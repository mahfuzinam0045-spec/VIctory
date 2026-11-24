//! The console's audio DSP (`chad.dll`): streams that convolve blocks of
//! float samples with an impulse response (reverb), or run other block
//! transforms, on a coprocessor.
//!
//! A title connects once with a memory block for the DSP, activates
//! streams, and submits one command per block (gains, an input and an
//! output buffer). Activation hands back a status block whose second word is
//! the ticket of the last completed command and whose fourth is its result;
//! the title checks it one block later and treats a late command as an
//! overrun. Commands complete when submitted here.
//!
//! The transforms themselves are not computed: the hardware takes impulse
//! responses already in its own partitioned frequency-domain layout, which
//! is not known. Outputs are silence by default, or an algorithmic reverb
//! fallback when `VICTORY_AUDIO_REVERB=1` is set.
// shortcut: hardware convolution IR layout is not known; outputs are zeros
// by default, with an algorithmic Schroeder reverb fallback when enabled.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::sync::Mutex;

use super::xasync::E_INVALIDARG;
use super::xgameruntime::{HRESULT, S_OK};
use crate::{HleModule, hle_export};

const E_POINTER: HRESULT = 0x8000_4003_u32 as i32;
const E_OUTOFMEMORY: HRESULT = 0x8007_000E_u32 as i32;

const CONVOLUTION: u32 = 1;
/// Option bits that need a stereo stream, and one convolution refuses.
const STEREO_OPTIONS: u32 = 2 | 8;
const NOT_FOR_CONVOLUTION: u32 = 4;

pub struct Dsp {
    max_streams: u32,
    active: Mutex<u32>,
}

/// What activation describes.
#[repr(C, packed)]
pub struct StreamDesc {
    kind: u32,
    options: u32,
    frames: u32,
    channels: u32,
    ir_floats: u32,
    ir: *const f32,
}

/// The status block a stream's owner polls.
#[repr(C)]
pub struct Status {
    submitted: u32,
    completed: u32,
    reserved: u32,
    result: HRESULT,
}

/// Comb filter for the algorithmic reverb fallback.
#[derive(Debug)]
struct Comb {
    buffer: Vec<f32>,
    index: usize,
    feedback: f32,
    damp: f32,
    store: f32,
}

impl Comb {
    fn new(delay: usize, feedback: f32, damp: f32) -> Self {
        Self {
            buffer: vec![0.0; delay],
            index: 0,
            feedback,
            damp,
            store: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        let output = self.buffer[self.index];
        self.store = output * (1.0 - self.damp) + self.store * self.damp;
        self.buffer[self.index] = input + self.store * self.feedback;
        self.index += 1;
        if self.index >= self.buffer.len() {
            self.index = 0;
        }
        output
    }
}

/// Allpass filter for diffuse room reflections.
#[derive(Debug)]
struct Allpass {
    buffer: Vec<f32>,
    index: usize,
    feedback: f32,
}

impl Allpass {
    fn new(delay: usize, feedback: f32) -> Self {
        Self {
            buffer: vec![0.0; delay],
            index: 0,
            feedback,
        }
    }

    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        let buf_out = self.buffer[self.index];
        let output = -input + buf_out;
        self.buffer[self.index] = input + buf_out * self.feedback;
        self.index += 1;
        if self.index >= self.buffer.len() {
            self.index = 0;
        }
        output
    }
}

/// Schroeder-style algorithmic reverb generator used as fallback when hardware
/// frequency-domain convolution impulse responses cannot be evaluated.
#[derive(Debug)]
pub struct Reverb {
    combs_l: [Comb; 4],
    combs_r: [Comb; 4],
    allpasses_l: [Allpass; 2],
    allpasses_r: [Allpass; 2],
}

impl Default for Reverb {
    fn default() -> Self {
        Self::new()
    }
}

impl Reverb {
    pub fn new() -> Self {
        Self {
            combs_l: [
                Comb::new(1557, 0.84, 0.2),
                Comb::new(1617, 0.84, 0.2),
                Comb::new(1491, 0.84, 0.2),
                Comb::new(1422, 0.84, 0.2),
            ],
            combs_r: [
                Comb::new(1557 + 23, 0.84, 0.2),
                Comb::new(1617 + 23, 0.84, 0.2),
                Comb::new(1491 + 23, 0.84, 0.2),
                Comb::new(1422 + 23, 0.84, 0.2),
            ],
            allpasses_l: [
                Allpass::new(225, 0.5),
                Allpass::new(556, 0.5),
            ],
            allpasses_r: [
                Allpass::new(225 + 23, 0.5),
                Allpass::new(556 + 23, 0.5),
            ],
        }
    }

    pub fn process(&mut self, input: &[f32], output: &mut [f32], gains: [f32; 2], channels: u32) {
        let (in_gain, out_gain) = (gains[0], gains[1]);
        if channels == 1 {
            for (i, o) in input.iter().zip(output.iter_mut()) {
                let x = *i * in_gain;
                let mut comb_out = 0.0;
                for c in &mut self.combs_l {
                    comb_out += c.process(x);
                }
                let mut ap_out = comb_out * 0.25;
                for a in &mut self.allpasses_l {
                    ap_out = a.process(ap_out);
                }
                *o = ap_out * out_gain;
            }
        } else if channels == 2 {
            let (in_pairs, _) = input.as_chunks::<2>();
            let (out_pairs, _) = output.as_chunks_mut::<2>();
            for (i_chunk, o_chunk) in in_pairs.iter().zip(out_pairs.iter_mut()) {
                let xl = i_chunk[0] * in_gain;
                let xr = i_chunk[1] * in_gain;

                let mut out_l = 0.0;
                for c in &mut self.combs_l {
                    out_l += c.process(xl);
                }
                let mut ap_l = out_l * 0.25;
                for a in &mut self.allpasses_l {
                    ap_l = a.process(ap_l);
                }

                let mut out_r = 0.0;
                for c in &mut self.combs_r {
                    out_r += c.process(xr);
                }
                let mut ap_r = out_r * 0.25;
                for a in &mut self.allpasses_r {
                    ap_r = a.process(ap_r);
                }

                o_chunk[0] = ap_l * out_gain;
                o_chunk[1] = ap_r * out_gain;
            }
        }
    }
}

pub struct Stream {
    dsp: *mut Dsp,
    frames: u32,
    channels: u32,
    reverb: Option<Reverb>,
    status: Box<Status>,
}

/// One block of work.
#[repr(C)]
pub struct Command {
    gains: [f32; 2],
    input: *const f32,
    output: *mut f32,
    reserved: u64,
}

pub extern "system" fn XDspConnectWithMaximumStreamLimit(
    base: *mut c_void,
    base_length: u32,
    ir_seconds: u32,
    max_streams: u32,
    max_channels: u32,
    out: *mut *mut Dsp,
) -> HRESULT {
    if base.is_null()
        || !(0x1_0000..=0x1F40_0000).contains(&base_length)
        || ir_seconds > 0x80
        || max_streams > 0x100
        || max_channels > 2
    {
        return E_INVALIDARG;
    }
    if out.is_null() {
        return E_POINTER;
    }
    tracing::info!(
        base_length,
        ir_seconds,
        max_streams,
        max_channels,
        "audio DSP connected"
    );
    let dsp = Box::new(Dsp {
        max_streams,
        active: Mutex::new(0),
    });
    // SAFETY: checked non-null.
    unsafe { out.write(Box::into_raw(dsp)) };
    S_OK
}

pub extern "system" fn XDspDisconnect(dsp: *mut Dsp) -> HRESULT {
    if dsp.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: made by `XDspConnectWithMaximumStreamLimit`; disconnected
    // once, after its streams.
    drop(unsafe { Box::from_raw(dsp) });
    S_OK
}

fn valid(desc: &StreamDesc) -> bool {
    let (kind, options, frames, channels) = (desc.kind, desc.options, desc.frames, desc.channels);
    let shape = if kind == CONVOLUTION {
        (1..=2).contains(&channels)
            && matches!(frames, 0x200 | 0x400)
            && options & NOT_FOR_CONVOLUTION == 0
    } else {
        channels == 1 && matches!(frames, 0x200 | 0x400 | 0x800)
    };
    shape && (options & STEREO_OPTIONS == 0 || channels == 2)
}

pub extern "system" fn XDspActivate(
    dsp: *mut Dsp,
    desc: *const StreamDesc,
    status: *mut *const Status,
    stream: *mut *mut Stream,
) -> HRESULT {
    if status.is_null() || stream.is_null() {
        return E_POINTER;
    }
    if dsp.is_null() || desc.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null; the title passes its description.
    let desc = unsafe { &*desc };
    if !valid(desc) {
        return E_INVALIDARG;
    }
    let (kind, options, frames, channels) = (desc.kind, desc.options, desc.frames, desc.channels);
    tracing::debug!(
        kind,
        options,
        frames,
        channels,
        "audio DSP stream activated"
    );
    {
        // SAFETY: a connection from `XDspConnectWithMaximumStreamLimit`.
        let dsp = unsafe { &*dsp };
        let mut active = dsp.active.lock().unwrap_or_else(|e| e.into_inner());
        if *active >= dsp.max_streams {
            return E_OUTOFMEMORY;
        }
        *active += 1;
    }
    let reverb_enabled = std::env::var("VICTORY_AUDIO_REVERB")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let reverb = if kind == CONVOLUTION && reverb_enabled {
        Some(Reverb::new())
    } else {
        None
    };
    let new = Box::new(Stream {
        dsp,
        frames: desc.frames,
        channels: desc.channels,
        reverb,
        status: Box::new(Status {
            submitted: 0,
            completed: 0,
            reserved: 0,
            result: S_OK,
        }),
    });
    // SAFETY: checked non-null.
    unsafe {
        status.write(&*new.status);
        stream.write(Box::into_raw(new));
    }
    S_OK
}

pub extern "system" fn XDspDeactivate(stream: *mut Stream) -> HRESULT {
    if stream.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: made by `XDspActivate`; deactivated once.
    let stream = unsafe { Box::from_raw(stream) };
    // SAFETY: the connection outlives its streams.
    let dsp = unsafe { &*stream.dsp };
    *dsp.active.lock().unwrap_or_else(|e| e.into_inner()) -= 1;
    S_OK
}

/// Completes the command at once: its output block is silence.
pub extern "system" fn XDspSubmitCommand(
    stream: *mut Stream,
    command: *const Command,
    ticket: *mut u32,
) -> HRESULT {
    if stream.is_null() || command.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: a stream from `XDspActivate`, the title's command.
    let (stream, command) = unsafe { (&mut *stream, &*command) };
    if !command.output.is_null() {
        let samples = (stream.frames * stream.channels) as usize;
        if let (Some(reverb), false) = (&mut stream.reverb, command.input.is_null()) {
            // SAFETY: title passes an input buffer holding `samples` floats.
            let input = unsafe { std::slice::from_raw_parts(command.input, samples) };
            // SAFETY: title passes an output buffer holding `samples` floats.
            let output = unsafe { std::slice::from_raw_parts_mut(command.output, samples) };
            reverb.process(input, output, command.gains, stream.channels);
        } else {
            // Default: silence output.
            // SAFETY: the title's output buffer holds one block of the stream.
            unsafe { std::ptr::write_bytes(command.output, 0, samples) };
        }
    }
    let status = &mut *stream.status;
    let next = status.submitted.wrapping_add(1);
    // SAFETY: the status block is read by title code on other threads.
    unsafe {
        std::ptr::write_volatile(&mut status.submitted, next);
        std::ptr::write_volatile(&mut status.result, S_OK);
        std::ptr::write_volatile(&mut status.completed, next);
    }
    if !ticket.is_null() {
        // SAFETY: optional out parameter.
        unsafe { ticket.write(next) };
    }
    S_OK
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "dsp",
        dlls: &["chad.dll"],
        exports: vec![
            hle_export!(XDspConnectWithMaximumStreamLimit),
            hle_export!(XDspDisconnect),
            hle_export!(XDspActivate),
            hle_export!(XDspDeactivate),
            hle_export!(XDspSubmitCommand),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolution_stream_as_the_title_drives_it() {
        let mut base = vec![0u8; 0x10_0000];
        let mut dsp = std::ptr::null_mut();
        assert_eq!(
            XDspConnectWithMaximumStreamLimit(
                base.as_mut_ptr().cast(),
                0x10_0000,
                24,
                0x80,
                1,
                &mut dsp
            ),
            S_OK
        );
        let ir = vec![0f32; 0x800];
        // Stereo convolution with option 2, as the title asks for.
        let desc = StreamDesc {
            kind: CONVOLUTION,
            options: 2,
            frames: 0x400,
            channels: 2,
            ir_floats: 0x800,
            ir: ir.as_ptr(),
        };
        let (mut status, mut stream) = (std::ptr::null(), std::ptr::null_mut());
        assert_eq!(XDspActivate(dsp, &desc, &mut status, &mut stream), S_OK);
        let mut output = vec![1f32; 0x800];
        let input = vec![0.5f32; 0x800];
        let command = Command {
            gains: [1.0, 1.0],
            input: input.as_ptr(),
            output: output.as_mut_ptr(),
            reserved: 0,
        };
        let mut ticket = 0;
        assert_eq!(XDspSubmitCommand(stream, &command, &mut ticket), S_OK);
        // SAFETY: the stream's status block.
        let (completed, result) = unsafe { ((*status).completed, (*status).result) };
        assert_eq!((completed, result), (ticket, S_OK));
        assert!(output.iter().all(|&s| s == 0.0));
        assert_eq!(XDspDeactivate(stream), S_OK);

        // Mono streams refuse stereo options; transforms are mono only.
        let bad = StreamDesc {
            channels: 1,
            ..desc
        };
        assert_eq!(
            XDspActivate(dsp, &bad, &mut status, &mut stream),
            E_INVALIDARG
        );
        let transform = StreamDesc {
            kind: 2,
            options: 0,
            channels: 1,
            ir_floats: 0,
            ir: std::ptr::null(),
            ..desc
        };
        assert_eq!(
            XDspActivate(dsp, &transform, &mut status, &mut stream),
            S_OK
        );
        assert_eq!(XDspDeactivate(stream), S_OK);
        assert_eq!(XDspDisconnect(dsp), S_OK);
    }

    #[test]
    fn reverb_fallback_when_enabled() {
        let mut reverb = Reverb::new();
        let mut input = vec![0.0f32; 1024];
        input[0] = 1.0;
        input[1] = 1.0;
        let mut output = vec![0.0f32; 1024];
        reverb.process(&input, &mut output, [1.0, 1.0], 2);
        assert!(output.iter().any(|&s| s != 0.0));
    }
}
