__device__
float det3(
    float a0, float a1, float a2,
    float a3, float a4, float a5,
    float a6, float a7, float a8
)
{
    return ((a0 * a4 * a8 + a1 * a5 * a6 + a2 * a3 * a7)
            - (a6 * a4 * a2 + a7 * a5 * a0 + a8 * a3 * a1));
}


__device__
void fit_plane_lsq(int *samples, float *v_map, float *u_map, float *disp,
                   int sample_nr, float *dv, float *du, float *d)
{
    float x1 = 0, x2 = 0, y1 = 0, y2 = 0, z1 = 0, xz = 0, yz = 0, xy = 0, r = 0;
    for(int i = 0; i < sample_nr; i++) {
        int idx = samples[i];
        float x = v_map[idx], y = u_map[idx], d = disp[idx];

        x1 += x;
        x2 += x * x;
        xz += x * d;

        y1 += y;
        y2 += y * y;
        yz += y * d;

        z1 += d;
        xy += x * y;
    }
    r  = det3(x2, xy, x1, xy, y2, y1, x1, y1, sample_nr) + 1e-15;

    *dv = det3(xz, xy, x1, yz, y2, y1, z1, y1, sample_nr) / r;
    *du = det3(x2, xz, x1, xy, yz, y1, x1, z1, sample_nr) / r;
    *d  = det3(x2, xy, xz, xy, y2, yz, x1, y1, z1) / r;
}

__device__
void ransac_fit_plane(float *v_map, float *u_map, float *disp, float *coef, int samples, int *sample_list, int iters, int sample_nr, float disp_th)
{
    int max_inliers = 0;
    float best_dv = 0.0;
    float best_du = 0.0;
    float best_d  = 0.0;

    for (int iter = 0; iter < iters; ++iter)
    {
        float dv = 0, du=0, d=0;
        fit_plane_lsq(sample_list + iter * sample_nr, v_map, u_map, disp, sample_nr, &dv, &du, &d);

        int inliers = 0;
        for (int i = 0; i < samples; ++i)  {
            float err = fabsf(dv * v_map[i] + du * u_map[i] + d - disp[i]);
            if (err < disp_th)
                inliers++;
        }

        if (inliers >= max_inliers) {
            max_inliers = inliers;
            best_d = d;
            best_dv = dv;
            best_du = du;
        }
    }
    coef[0] = best_d;
    coef[1] = best_dv;
    coef[2] = best_du;
}

extern "C" {

__global__ void ransac_fit_plane_kernel(
    float *v_map, float *u_map, float *data,
    float *coeff, int pixels, int sample_size,
    int *sample_list, int iters, int sample_nr, float disp_th)
{
  unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int r = row; r < pixels; r += stride)
    ransac_fit_plane(v_map, u_map, data + sample_size * r, coeff + 3*r, sample_size, sample_list, iters, sample_nr, disp_th);
}

} // extern "C"
