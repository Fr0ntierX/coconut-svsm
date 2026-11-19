// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright (c) 2026 Fr0ntierX
//
// Fault injection for the retire and watchdog error paths (F0X-75).
//
// Most of those paths need the host to refuse a request or the hardware to
// return a status it never returns on the lab part, so they cannot be reached
// from a guest. This module lets a parttest guest arm one failure at a named
// step, and the step's call site then fails the way the real failure would:
//
// * AP_DESTROY and AP_CREATE are sent to the real host with an invalid VMPL,
//   which KVM refuses before it looks at the target. The SVSM's handling of
//   the host's answer is exercised as it is.
// * RMPADJUST failures and mapping failures replace the operation, so the RMP
//   is left exactly as a real failure would leave it.
//
// The whole module, and every call site through `fault_inject!`, exists only
// with the `parttest` feature. A build without it has no fault state and no
// injected branches.

use crate::locking::SpinLock;

/// A VMPL KVM refuses in every AP Creation request (it checks the VMPL is
/// below 4 before anything else). The GHCB helpers keep 4 bits of it.
pub(crate) const FAULT_BAD_VMPL: u64 = 0xf;

/// Where a failure can be injected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub(crate) enum FaultStep {
    /// The retire's AP_DESTROY goes to the host with an invalid VMPL.
    ApDestroy = 1,
    /// Mapping the VMSA page in the retire fails.
    VmsaMap = 2,
    /// The retire this hit lands in sees the armed value's worth of
    /// FAIL_INUSE results from the VMSA RMPADJUST, without it running, and
    /// then the real one. u64::MAX keeps failing until the deadline. Every
    /// retire is one hit, so `skip` picks the retire.
    VmsaInuse = 3,
    /// RMPADJUST on the VMSA returns the armed value without running.
    VmsaAdjust = 4,
    /// Revoking a partition page returns the armed value without running.
    /// Every page is one hit, so `skip` picks the page index.
    PageRevoke = 5,
    /// The first RMPADJUST that hands the old VMSA page back (VMPL3 none)
    /// returns the armed value without running.
    ReleaseVmpl3 = 6,
    /// The old VMSA page looks read-only in the host's mapping, so it is not
    /// scrubbed.
    ReleaseReadOnly = 7,
    /// The RMPADJUST that gives the scrubbed VMSA page back to the guest VMPL
    /// returns the armed value without running.
    ReleaseGrant = 8,
    /// START_VCPU's AP_CREATE goes to the host with an invalid VMPL.
    ApCreate = 9,
    /// A watchdog heartbeat read fails. Every read is one hit.
    HeartbeatRead = 10,
    /// Arming a watchdog slot finds no free slot.
    WatchdogFull = 11,
}

impl FaultStep {
    pub(crate) fn from_code(code: u64) -> Option<Self> {
        Some(match code {
            1 => Self::ApDestroy,
            2 => Self::VmsaMap,
            3 => Self::VmsaInuse,
            4 => Self::VmsaAdjust,
            5 => Self::PageRevoke,
            6 => Self::ReleaseVmpl3,
            7 => Self::ReleaseReadOnly,
            8 => Self::ReleaseGrant,
            9 => Self::ApCreate,
            10 => Self::HeartbeatRead,
            11 => Self::WatchdogFull,
            _ => return None,
        })
    }
}

/// One armed failure.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Fault {
    pub step: FaultStep,
    /// What the failing step returns, where it returns a status.
    pub value: u64,
    /// Hits at the step to let through before the first failure.
    pub skip: u32,
    /// Failures left. The fault disarms itself at zero.
    pub count: u32,
}

/// What `fault_arm` replaced, for the guest to confirm a fault fired.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct FaultReport {
    /// Step of the previous fault, 0 if none was ever armed.
    pub step: u64,
    /// Times it fired.
    pub fired: u64,
    /// Failures it had left.
    pub remaining: u32,
}

struct FaultState {
    armed: Option<Fault>,
    last_step: u64,
    fired: u64,
}

static FAULT: SpinLock<FaultState> = SpinLock::new(FaultState {
    armed: None,
    last_step: 0,
    fired: 0,
});

/// Arm `fault`, or clear it with `None`, and report the one it replaces.
pub(crate) fn fault_arm(fault: Option<Fault>) -> FaultReport {
    let mut st = FAULT.lock();
    let report = FaultReport {
        step: st.last_step,
        fired: st.fired,
        remaining: st.armed.map_or(0, |f| f.count),
    };
    st.armed = fault.filter(|f| f.count > 0);
    st.last_step = fault.map_or(0, |f| f.step as u64);
    st.fired = 0;
    report
}

/// Called at `step`. Returns the armed value when this hit should fail.
pub(crate) fn fault_hit(step: FaultStep) -> Option<u64> {
    let mut st = FAULT.lock();
    let f = st.armed.as_mut().filter(|f| f.step == step)?;
    if f.skip > 0 {
        f.skip -= 1;
        return None;
    }
    f.count -= 1;
    let value = f.value;
    if f.count == 0 {
        st.armed = None;
    }
    st.fired += 1;
    Some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    // One test touches the global, so the cases cannot race each other.
    #[test]
    fn arm_skip_count_and_report() {
        fault_arm(None);
        assert_eq!(fault_hit(FaultStep::ApDestroy), None);

        let prev = fault_arm(Some(Fault {
            step: FaultStep::PageRevoke,
            value: 6,
            skip: 2,
            count: 2,
        }));
        assert_eq!(prev.step, 0);
        assert_eq!(fault_hit(FaultStep::VmsaAdjust), None);
        assert_eq!(fault_hit(FaultStep::PageRevoke), None);
        assert_eq!(fault_hit(FaultStep::PageRevoke), None);
        assert_eq!(fault_hit(FaultStep::PageRevoke), Some(6));
        assert_eq!(fault_hit(FaultStep::PageRevoke), Some(6));
        assert_eq!(fault_hit(FaultStep::PageRevoke), None);

        let prev = fault_arm(Some(Fault {
            step: FaultStep::VmsaInuse,
            value: 0,
            skip: 0,
            count: 5,
        }));
        assert_eq!(prev.step, FaultStep::PageRevoke as u64);
        assert_eq!(prev.fired, 2);
        assert_eq!(prev.remaining, 0);
        assert_eq!(fault_hit(FaultStep::VmsaInuse), Some(0));

        let prev = fault_arm(None);
        assert_eq!(prev.fired, 1);
        assert_eq!(prev.remaining, 4);
        assert_eq!(fault_hit(FaultStep::VmsaInuse), None);
        assert_eq!(FaultStep::from_code(12), None);
        assert_eq!(FaultStep::from_code(5), Some(FaultStep::PageRevoke));
    }
}
