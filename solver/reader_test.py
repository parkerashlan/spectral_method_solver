from solver.io_hdf5 import RunReader

with RunReader(
    "/home/ashlan/physics/research/code/spectral_method_solver/outputs/alfvenwave_outputs/2026-09-07/23-53-14/run.h5"
) as r:
    print("completed", r.attrs["completed"])
    print("skipped  ", r.attrs["skipped_params"])
    print("snaps", r.n_snapshots, "t", r.t[:])
    print("A  ", r.A[0].min(), r.A[0].max())
    print("psi", r.psi[0].min(), r.psi[0].max())
    print("e_kin", r.scalar("e_kin")[:4])
    print("jmax ", r.scalar("jmax")[:4])
