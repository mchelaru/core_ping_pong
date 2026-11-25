use core_affinity::{CoreId, get_core_ids, set_for_current};
use std::hint::spin_loop;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

const ROUNDS: usize = 50_000;
const WARMUP: u64 = 1_000;
const CACHE_LINE: usize = 128;

/// One flag per cache line; each side only ever writes its own line.
#[repr(C, align(64))]
struct Flag {
    value: AtomicU64,
    _pad: [u8; CACHE_LINE - 8],
}

impl Flag {
    const fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
            _pad: [0; CACHE_LINE - 8],
        }
    }
}

fn pin_to_cpu(cpu: &CoreId) {
    assert!(set_for_current(*cpu));
}

#[cfg(not(target_os = "netbsd"))]
fn system_tsc_hz() -> anyhow::Result<u64> {
    raw_cpuid::CpuId::new()
        .get_tsc_info()
        .ok_or(anyhow::anyhow!("Unable to get tsc info"))?
        .tsc_frequency()
        .ok_or(anyhow::anyhow!("Unable to get tsc frequency"))
}

#[cfg(target_os = "netbsd")]
fn system_tsc_hz() -> anyhow::Result<u64> {
    let mut len = size_of::<u64>();
    let mut hz: u64 = 0;
    let name = std::ffi::CString::new("machdep.tsc_freq")?;
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &mut hz as *mut u64 as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 {
        Ok(hz)
    } else {
        Err(anyhow::anyhow!("sysctl machdep.tsc_freq failed"))
    }
}

/// Reads the TSC and the monotonic clock together; the TSC value is the
/// midpoint of two reads bracketing `Instant::now()`.
fn tsc_and_instant() -> (u64, Instant) {
    let before = rdtsc_start();
    let now = Instant::now();
    let after = rdtsc_stop();
    (before + (after - before) / 2, now)
}

/// Estimates the TSC frequency by busy-spinning over several wall-clock
/// intervals and fitting a least-squares line through (elapsed ns, elapsed ticks).
fn calibrate_tsc_hz() -> anyhow::Result<u64> {
    const INTERVALS_MS: [u64; 5] = [10, 20, 50, 100, 200];
    const REPEATS: usize = 3;

    let mut points: Vec<(f64, f64)> = Vec::with_capacity(INTERVALS_MS.len() * REPEATS);
    for _ in 0..REPEATS {
        for &ms in &INTERVALS_MS {
            let target = Duration::from_millis(ms);
            let (tsc0, t0) = tsc_and_instant();
            let (tsc1, t1) = loop {
                let (tsc, t) = tsc_and_instant();
                if t.duration_since(t0) >= target {
                    break (tsc, t);
                }
                spin_loop();
            };
            points.push((
                t1.duration_since(t0).as_nanos() as f64,
                tsc1.wrapping_sub(tsc0) as f64,
            ));
        }
    }

    let n = points.len() as f64;
    let mean_x = points.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y = points.iter().map(|p| p.1).sum::<f64>() / n;
    let (cov, var) = points.iter().fold((0.0, 0.0), |(cov, var), &(x, y)| {
        let dx = x - mean_x;
        (cov + dx * (y - mean_y), var + dx * dx)
    });
    let ticks_per_ns = cov / var;
    if !ticks_per_ns.is_finite() || ticks_per_ns <= 0.0 {
        return Err(anyhow::anyhow!("TSC calibration produced an invalid slope"));
    }
    Ok((ticks_per_ns * 1e9).round() as u64)
}

#[inline(always)]
fn rdtsc_start() -> u64 {
    unsafe {
        core::arch::x86_64::_mm_lfence();
        core::arch::x86_64::_rdtsc()
    }
}

#[inline(always)]
fn rdtsc_stop() -> u64 {
    unsafe {
        let mut aux = 0u32;
        let v = core::arch::x86_64::__rdtscp(&mut aux);
        core::arch::x86_64::__cpuid(0);
        v
    }
}

fn cycles_to_ns(cycles: u64, hz: u64) -> u64 {
    ((cycles as u128) * 1_000_000_000u128 / hz as u128) as u64
}

fn main() -> anyhow::Result<()> {
    let ping_flag = Flag::new();
    let pong_flag = Flag::new();

    let cores = get_core_ids().ok_or(anyhow::anyhow!("Unable to get core set"))?;
    let ping_core = cores
        .get(1)
        .ok_or(anyhow::anyhow!("Unable to get first core"))?;
    let pong_core = cores
        .get(2)
        .ok_or(anyhow::anyhow!("Unable to get second core"))?;

    let hz = match system_tsc_hz() {
        Ok(hz) if hz > 0 => hz,
        result => {
            if let Err(e) = result {
                eprintln!("{e}; calibrating TSC frequency");
            }
            pin_to_cpu(cores.get(0).ok_or(anyhow::anyhow!(
                "Unable to pin core in order to measure tsc"
            ))?);
            let hz = calibrate_tsc_hz()?;
            eprintln!("Calibrated TSC frequency: {hz} Hz");
            hz
        }
    };

    let mut samples = thread::scope(|s| {
        let t_ping = s.spawn(|| {
            pin_to_cpu(ping_core);
            ping(&ping_flag, &pong_flag)
        });
        let t_pong = s.spawn(|| {
            pin_to_cpu(pong_core);
            pong(&ping_flag, &pong_flag)
        });
        let samples = t_ping.join().expect("ping join");
        t_pong.join().expect("pong");
        samples
    });

    samples.sort_unstable();
    println!("Median: {}ns", cycles_to_ns(samples[ROUNDS / 2], hz));

    Ok(())
}

fn ping(ping_flag: &Flag, pong_flag: &Flag) -> Vec<u64> {
    let mut samples = Vec::with_capacity(ROUNDS);

    for seq in 1..=WARMUP {
        ping_flag.value.store(seq, Ordering::Release);
        while pong_flag.value.load(Ordering::Acquire) != seq {
            spin_loop();
        }
    }

    let first = WARMUP + 1;
    let last = WARMUP + ROUNDS as u64;
    for seq in first..=last {
        let start = rdtsc_start();
        ping_flag.value.store(seq, Ordering::Release);
        while pong_flag.value.load(Ordering::Acquire) != seq {
            spin_loop();
        }
        let stop = rdtsc_stop();
        samples.push(stop - start);
    }

    samples
}

fn pong(ping_flag: &Flag, pong_flag: &Flag) {
    let total = WARMUP + ROUNDS as u64;
    for seq in 1..=total {
        while ping_flag.value.load(Ordering::Acquire) != seq {
            spin_loop();
        }
        pong_flag.value.store(seq, Ordering::Release);
    }
}
