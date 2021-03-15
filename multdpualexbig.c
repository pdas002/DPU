#include <stdio.h>
#include <stdint.h>
#include <alloc.h>
#include <mram.h>
#include <perfcounter.h>
#include <float.h>

#define BUFFER_SIZE 489216
#define LBUFFER_SIZE 624
#define DIGIT_SIZE 3136
#define MAX_READ 2048


#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)
#define CEIL_POS(X) ((X-(int)(X)) > 0 ? (int)(X+1) : (int)(X))
#define MAX_FILTER_BYTES 12
const uint8_t bits[8] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};


__mram_noinit uint8_t inputdata[BUFFER_SIZE];
__mram_noinit uint8_t labeldata[LBUFFER_SIZE];
__host uint32_t total;


/* layer types */
 void blinear_layer(const uint8_t* A, const uint8_t* F, uint8_t* C,
                          const float* Bias, const float* Gamma,
                          const float* Beta, const float* Mean,
                          const float* Std, const int m, const int n,
                          const int k);
 void fconv_layer(const float* A, const uint8_t* F, uint8_t* C,
                        const float* Bias, const float* Gamma, const float* Beta,
                        const float* Mean, const float* Std, const int m,
                        const int num_f, const int w, const int h, const int d,
                        const int kw, const int kh, const int sw, const int sh,
                        const int pw, const int ph, const int pl_w,
                        const int pl_h, const int pl_sw, const int pl_sh,
                        const int pl_pw, const int pl_ph);
 void bconv_layer(const uint8_t* A, const uint8_t* F, uint8_t* C,
                        const float* Bias, const float* Gamma, const float* Beta,
                        const float* Mean, const float* Std, const int m,
                        const int num_f, const int w, const int h, const int d,
                        const int kw, const int kh, const int sw, const int sh,
                        const int pw, const int ph, const int pl_w,
                        const int pl_h, const int pl_sw, const int pl_sh,
                        const int pl_pw, const int pl_ph);

/* layer helper functions */
 float batch_norm(float f, const float Gamma, const float Beta,
                        const float Mean, const float Std);
 void fconv(const float* A, const uint8_t* F, uint8_t* C,
                  const int c_start_idx, const float Bias, const float Gamma,
                  const float Beta, const float Mean, const float Std,
                  const int w, const int h, const int d, const int kw,
                  const int kh, const int sw, const int sh, const int pw,
                  const int ph, const int pl_w, const int pl_h, const int pl_sw,
                  const int pl_sh, const int pl_pw, const int pl_ph);
 void bconv(const uint8_t* A, const uint8_t* F, uint8_t* C,
                  const int c_start_idx, const int z, const float Bias,
                  const float Gamma, const float Beta, const float Mean,
                  const float Std, const int w, const int h, const int d,
                  const int kw, const int kh, const int sw, const int sh,
                  const int pw, const int ph, const int pl_w, const int pl_h,
                  const int pl_sw, const int pl_sh, const int pl_pw,
                  const int pl_ph);
 float fdot_3d(const float* A, const uint8_t* B, const int x, const int y,
                     const int w, const int h, const int d, const int kw,
                     const int kh);
 int bdot_3d(const uint8_t* A, const uint8_t* B, const int x, const int y,
                   const int z, const int w, const int h, const int d,
                   const int kw, const int kh);
 int bdot(const uint8_t* A, const uint8_t* B, const int N);

/* indexing functions */
 int idx_2d(const int i, const int j, const int rows);
 int idx_3d(const int i, const int j, const int k, const int rows,
                  const int cols);
 int idx_4d(const int i, const int j, const int k, const int l,
                  const int rows, const int cols, const int depth);
 int conv_idx(const int pl_i, const int x, const int kx, const int sx,
                    const int px);
 int convpool_size(const int x, const int kx, const int sx, const int px,
                         const int pl_x, const int pl_sx, const int pl_px);

/* Bit functions */
 uint8_t rotr1 (const uint8_t x);
 int popcnt8(const uint8_t v);
 int nthbitset_arr(const uint8_t* const arr, const int n);
 int bslice_2d(uint8_t* const dst, const uint8_t* const src, const int x,
                     const int y, const int w, const int h, const int kw,
                     const int kh);
 int bslice_2d_filter(uint8_t* const dst, const uint8_t* const src,
                            const int x, const int y, const int w, const int h,
                            const int kw, const int kh);
 int bslice_4d(uint8_t* const dst, const uint8_t* const src, const int x,
                     const int y, const int zi, const int zj, const int w,
                     const int h, const int d, const int kw, const int kh);

/* layers types */
void blinear_layer(const uint8_t* A, const uint8_t* F, uint8_t* C,
                          const float* Bias, const float* Gamma,
                          const float* Beta, const float* Mean,
                          const float* Std, const int m, const int n,
                          const int k)
{
  uint8_t c_mask, res_sign;
  int i, j, ni, ci, c_shift, c_idx;
  const uint8_t *Ari;
  float res;

  c_shift = 7;
  c_mask = 1 << c_shift;
  c_idx = 0;
 
  /* Compute ceil in terms of 8-bits strides */
  ni = (n + 7) / 8;
  for (i = 0; i < m; ++i) {
    Ari = A + (i * ni);
    for (j = 0; j < k; ++j) {
      ci = j * ni;
      res = bdot(Ari, F + ci, n);
      res += Bias[j];
      res = batch_norm(res, Gamma[j], Beta[j], Mean[j], Std[j]);
      res_sign = res >= 0 ? 1 : 0;

      /* store result */
      C[c_idx] |= res_sign << c_shift;

      /* update c_idx */
      c_mask = rotr1(c_mask);
      c_idx += (c_mask & 0x80) >> 7;
      c_shift--;
      c_shift =  c_shift < 0 ? 7 : c_shift;
    }
    c_shift = 7;
    c_mask = 1 << c_shift;
    c_idx += 1;
  }
}

void blinear_sm_layer(const uint8_t* A, const uint8_t* F, uint8_t* C,
                             const float* Bias, const float* Gamma,
                             const float* Beta, const float* Mean,
                             const float* Std, const int m, const int n,
                             const int k)
{
  int i, j, ni, ci,  max_idx;
  const uint8_t *Ari;
  float res, max_res;

  /* Compute ceil in terms of 8-bits strides */
  ni = (n + 7) / 8;
  for (i = 0; i < m; ++i) {
    max_res = -FLT_MAX;
    Ari = A + (i * ni);
    for (j = 0; j < k; ++j) {
      ci = j * ni;
      res = bdot(Ari, F + ci, n);
      res += Bias[j];
      res = batch_norm(res, Gamma[j], Beta[j], Mean[j], Std[j]);
      if (res > max_res) {
        max_idx = j;
        max_res = res;
      }
    }
    C[i] = max_idx;
  }
}

void fconv_layer(const float* A, const uint8_t* F, uint8_t* C,
                        const float* Bias, const float* Gamma, const float* Beta,
                        const float* Mean, const float* Std, const int m,
                        const int num_f, const int w, const int h, const int d,
                        const int kw, const int kh, const int sw, const int sh,
                        const int pw, const int ph, const int pl_w,
                        const int pl_h, const int pl_sw, const int pl_sh,
                        const int pl_pw, const int pl_ph)
{
  int i, j, max_m, res_size, res_w, res_h, c_idx, a_idx, f_idx, whd, cp_kwkhd;

  c_idx = 0;
  res_w = convpool_size(w, kw, sw, pw, pl_w, pl_sw, pl_pw);
  res_h = convpool_size(h, kh, sh, ph, pl_h, pl_sh, pl_ph);
  res_size = res_w * res_h;
  max_m = CEIL_POS(res_size*m*num_f/8.0);
  whd = w*h*d;
  cp_kwkhd = CEIL_POS(kw*kh*d/8.0);

  /* initialize result */
  for (i = 0; i < max_m; ++i) {
    C[i] = 0;
  }

  for (i = 0; i < m; ++i) {
    for (j = 0; j < num_f; ++j) {
      a_idx = i*whd;
      f_idx = j*cp_kwkhd;
      fconv(A + a_idx, F + f_idx, C, c_idx, Bias[j], Gamma[j],
            Beta[j], Mean[j], Std[j], w, h, d, kw, kh, sw, sh, pw, ph,
            pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);
      c_idx += res_size;
    }
  }
}

void bconv_layer(const uint8_t* A, const uint8_t* F, uint8_t* C,
                        const float* Bias, const float* Gamma, const float* Beta,
                        const float* Mean, const float* Std, const int m,
                        const int num_f, const int w, const int h, const int d,
                        const int kw, const int kh, const int sw, const int sh,
                        const int pw, const int ph, const int pl_w,
                        const int pl_h, const int pl_sw, const int pl_sh,
                        const int pl_pw, const int pl_ph)
{
  int i, j, max_m, res_size, res_w, res_h, c_idx, f_idx, dcp_kwkh;


  c_idx = 0;
  res_w = convpool_size(w, kw, sw, pw, pl_w, pl_sw, pl_pw);
  res_h = convpool_size(h, kh, sh, ph, pl_h, pl_sh, pl_ph);
  res_size = res_w * res_h;
  max_m = CEIL_POS(res_size*m*num_f/8.0);
  dcp_kwkh = d*CEIL_POS(kw*kh/8.0);

  /* initialize result */
  for (i = 0; i < max_m; ++i) {
    C[i] = 0;
  }

  for (i = 0; i < m; ++i) {
    for (j = 0; j < num_f; ++j) {
      f_idx = j*dcp_kwkh;
      bconv(A, F + f_idx, C, c_idx, i, Bias[j], Gamma[j],
            Beta[j], Mean[j], Std[j], w, h, d, kw, kh, sw, sh, pw, ph,
            pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);
      c_idx += res_size;
    }
  }
}

/* layer helper functions */
float batch_norm(float f, const float Gamma, const float Beta,
                        const float Mean, const float Std)
{
  f -= Mean;
  f /= Std;
  f *= Gamma;
  f += Beta;
  return f;
}

int bdot(const uint8_t* A, const uint8_t* B, const int N)
{
  int i, num_bytes, res;

  num_bytes = CEIL_POS(N/8.0);
  res = 0;
  for (i = 0; i < num_bytes; ++i) {
    res += popcnt8(~(A[i]^B[i]));
  }
  res = res*2 - N;
  return res;
}

int bdot_3d(const uint8_t* A, const uint8_t* B, const int x, const int y,
                    const int z, const int w, const int h, const int d,
                    const int kw, const int kh)
{
  /* Handles up to 10x10 filters */
  uint8_t A_slice[MAX_FILTER_BYTES] = {0};
  uint8_t B_slice[MAX_FILTER_BYTES] = {0};
  const uint8_t *B_idx;
  int i, comp_n, res, N, B_bytes, bx, by, bw, bh, w_x, h_y;

  N = kw*kh;
  B_bytes = CEIL_POS(kw*kh/8.0);
  res = 0;
  w_x = w - x;
  h_y = h - y;
  for (i = 0; i < d; ++i) {
    B_idx = B + B_bytes*i;
    comp_n = bslice_4d(A_slice, A, x, y, z, i, w, h, d, kw, kh);
    //no padding
    if (comp_n == N) {
      res += bdot(A_slice, B_idx, N);
    }
    //padding
    else {
      bx = -MIN(0, x);
      by = -MIN(0, y);
      bw = MIN(kw, w_x);
      bh = MIN(kh, h_y);
      bslice_2d_filter(B_slice, B_idx, bx, by, kw, kh, bw, bh);
      res += bdot(A_slice, B_slice, comp_n);
    }
  }

  return res;
}

/* float convolution + BN */
/* C_start_idx is the starting index for storing the result */
void fconv(const float* A, const uint8_t* F, uint8_t* C,
                  const int c_start_idx, const float Bias, const float Gamma,
                  const float Beta, const float Mean, const float Std,
                  const int w, const int h, const int d, const int kw,
                  const int kh, const int sw, const int sh, const int pw,
                  const int ph, const int pl_w, const int pl_h, const int pl_sw,
                  const int pl_sh, const int pl_pw, const int pl_ph)
{
  uint8_t c_mask, res_sign;
  int pl_i, pl_j, i, j, i_in, j_in, pl_i_max, pl_j_max, c_shift, c_idx, pl_w2_1, pl_hpw_1;
  float res, max_res;

  c_shift = 7 - (c_start_idx % 8);
  c_mask = 1 << c_shift;
  c_idx = c_start_idx / 8;
  pl_i_max = (w - kw + 2*pw)/sw + (2*pl_pw) + 1;
  pl_j_max = (h - kh + 2*ph)/sh + (2*pl_ph) + 1;
  pl_w2_1 = pl_w + pl_pw - 1;
  pl_hpw_1 = pl_h + pl_pw - 1;
  for (pl_i = -pl_pw; pl_i + pl_w2_1 < pl_i_max; pl_i += pl_sw) {
  for (pl_j = -pl_ph; pl_j + pl_hpw_1 < pl_j_max; pl_j += pl_sh) {
    max_res = res = -FLT_MAX;
    int pl_i_pl_w = pl_i + pl_w;
    for (i_in = pl_i; i_in < pl_i_pl_w; ++i_in) {
    i = conv_idx(i_in, w, kw, sw, pw);
    int pl_j_pl_h = pl_j + pl_h;
    for (j_in = pl_j; j_in < pl_j_pl_h; ++j_in) {
      j = conv_idx(j_in, h, kh, sh, ph);
      if (i >= -pw && j >= -ph) {
        res = fdot_3d(A, F, i, j, w, h, d, kw, kh);
        max_res = MAX(res, max_res);
      }
    }
    }
    max_res += Bias;
    max_res = batch_norm(max_res, Gamma, Beta, Mean, Std);
    res_sign = max_res >= 0 ? 1 : 0;

    /* store result */
    C[c_idx] |= res_sign << c_shift;

    /* update c_idx */
    c_mask = rotr1(c_mask);
    c_idx += (c_mask & 0x80) >> 7;
    c_shift--;
    c_shift =  c_shift < 0 ? 7 : c_shift;
  }
  }
}

void bconv(const uint8_t* A, const uint8_t* F, uint8_t* C,
                  const int c_start_idx, const int z, const float Bias,
                  const float Gamma, const float Beta, const float Mean,
                  const float Std, const int w, const int h, const int d,
                  const int kw, const int kh, const int sw, const int sh,
                  const int pw, const int ph, const int pl_w, const int pl_h,
                  const int pl_sw, const int pl_sh, const int pl_pw,
                  const int pl_ph)
{
  uint8_t c_mask, res_sign;
  int pl_i, pl_j, i, j, i_in, j_in, pl_i_max, pl_j_max, c_shift, c_idx, pl_w2_1, pl_hpw_1;
  float res, max_res;

  c_shift = 7 - (c_start_idx % 8);
  c_mask = 1 << c_shift;
  c_idx = c_start_idx / 8;
  pl_i_max = (w - kw + 2*pw)/sw + (2*pl_pw) + 1;
  pl_j_max = (h - kh + 2*ph)/sh + (2*pl_ph) + 1;
  pl_w2_1 = pl_w + pl_pw - 1;
  pl_hpw_1 = pl_h + pl_pw - 1;
  for (pl_i = -pl_pw; pl_i + pl_w2_1 < pl_i_max; pl_i += pl_sw) {
  for (pl_j = -pl_ph; pl_j + pl_hpw_1 < pl_j_max; pl_j += pl_sh) {
    max_res = res = -FLT_MAX;
    int pl_i_pl_w = pl_i + pl_w;
    for (i_in = pl_i; i_in < pl_i_pl_w; ++i_in) {
    i = conv_idx(i_in, w, kw, sw, pw);
    int pl_j_pl_h = pl_j + pl_h;
    for (j_in = pl_j; j_in < pl_j_pl_h; ++j_in) {
      j = conv_idx(j_in, h, kh, sh, ph);
      if (i >= -pw && j >= -ph) {
        res = bdot_3d(A, F, i, j, z, w, h, d, kw, kh);
        max_res = MAX(res, max_res);
      }
    }
    }
    max_res += Bias;
    max_res = batch_norm(max_res, Gamma, Beta, Mean, Std);
    res_sign = max_res >= 0 ? 1 : 0;

    /* store result */
    C[c_idx] |= res_sign << c_shift;

    /* update c_idx */
    c_mask = rotr1(c_mask);
    c_idx += (c_mask & 0x80) >> 7;
    c_shift--;
    c_shift =  c_shift < 0 ? 7 : c_shift;
  }
  }
  //printf("-");
}

float fdot_3d(const float* A, const uint8_t* B, const int x, const int y,
                     const int w, const int h, const int d, const int kw,
                     const int kh)
{
  uint8_t  bitset;
  int i, j, k, b_idx, A_bytes, x_kw, y_kh;
  float a, res;
  const float *A_slice;

  A_bytes = w*h;
  res = 0;
  b_idx = 0;
  x_kw = x + kw;
  y_kh = y + kh;
  for (i = 0; i < d; ++i) {
    A_slice = A + A_bytes*i;
    for (j = x; j < x_kw; ++j) {
      for (k = y; k < y_kh; ++k) {
        /* handles padding */
        if (j < 0 || j > h-1 || k < 0 || k > w-1) {
          a = 0.0;
        }
        else {
          a = A_slice[idx_2d(j, k, w)];
        }

        bitset = nthbitset_arr(B, b_idx);
        res += bitset ? a : -a;
        b_idx++;
      }
    }
  }

  return res;
}

/* index functions */
int idx_2d(const int i, const int j, const int rows)
{
  return i * rows + j;
}

int idx_3d(const int i, const int j, const int k, const int rows,
                  const int cols)
{
  return i * rows * cols + j * cols + k;
}

int idx_4d(const int i, const int j, const int k, const int l,
                  const int rows, const int cols, const int depth)
{
  return i * rows * cols * depth + j * cols * depth + k * depth + l;
}

int conv_idx(const int pl_i, const int x, const int kx, const int sx,
                    const int px)
{
  int conv_sz = (x - kx + 2*px)/sx;
  return (pl_i < 0 || pl_i > conv_sz) ? -INT_MAX : pl_i * sx - px;
}

int convpool_size(const int x, const int kx, const int sx, const int px,
                         const int pl_x, const int pl_sx, const int pl_px)
{
  return ((x - kx + 2*px)/sx - pl_x + (2*pl_px) + 1)/pl_sx + 1;
}

/* Bit functions */
uint8_t rotr1 (const uint8_t x)
{
  return (x >> 1) | (x << 7);
}

int nthbitset_arr(const uint8_t* const arr, const int n)
{
  return arr[n/8] & bits[n%8] ? 1 : 0;
}

int popcnt8(const uint8_t v) {
  uint8_t c;
  c = v - ((v >> 1) & 0x55);
  c = ((c >> 2) & 0x33) + (c & 0x33);
  return ((c >> 4) + c) & 0x0F;
}

int bslice_2d(uint8_t* const dst, const uint8_t* const src, const int x,
                     const int y, const int w, const int h, const int kw,
                     const int kh)
{
  int i, j, n, idx, shift, bytes, x_kw, y_kh;
  uint8_t mask, bitset;

  /* initiaize dst */
  bytes = CEIL_POS(kw*kh/8.0);
  for (i = 0; i < bytes; ++i) {
    dst[i] = 0;
  }

  idx = 0;
  shift = 7;
  n = 0;
  x_kw = x + kw;
  y_kh = y + kh;
  for (i = x; i < x_kw; ++i) {
    for (j = y; j < y_kh; ++j) {
      /* Padding out of bounds */
      if (i < 0 || i > h-1 || j < 0 || j > w-1) {
        continue;
      }

      bitset = nthbitset_arr(src, idx_2d(i, j, w));
      dst[idx/8] |= bitset << shift;

      mask = rotr1(mask);
      idx++;
      shift--;
      shift = shift < 0 ? 7 : shift;
      n++;
    }
  }

  return n;
}

int bslice_2d_filter(uint8_t* const dst, const uint8_t* const src,
                            const int x, const int y, const int w, const int h,
                            const int kw, const int kh)
{
  int i, j, n, idx, shift, bytes, bitset, x_kw, y_kh;
  uint8_t mask;

  /* initiaize dst */
  bytes = CEIL_POS(kw*kh/8.0);
  for (i = 0; i < bytes; ++i) {
    dst[i] = 0xFF;
  }

  idx = 0;
  shift = 7;
  n = 0;
  x_kw = x + kw;
  y_kh = y + kh;
  for (i = x; i < x_kw; ++i) {
    for (j = y; j < y_kh; ++j) {
      /* Padding out of bounds */
      if (i < 0 || i > h-1 || j < 0 || j > w-1) {
        continue;
      }

      bitset = nthbitset_arr(src, idx_2d(i, j, w));
      dst[idx/8] &= ~((!bitset) << shift);

      mask = rotr1(mask);
      idx++;
      shift--;
      shift = shift < 0 ? 7 : shift;
      n++;
    }
  }

  return n;
}

/* 4d slice function on binary matrix (bit packed) */
int bslice_4d(uint8_t* const dst, const uint8_t* const src, const int x,
                     const int y, const int zi, const int zj, const int w,
                     const int h, const int d, const int kw, const int kh)
{
  int i, j, n, idx, shift, bytes, bitset, x_kw, y_kh;
  uint8_t mask;

  /* initialize dest */
  bytes = CEIL_POS(kw*kh/8.0);
  for (i = 0; i < bytes; ++i) {
    dst[i] = 0;
  }

  idx = 0;
  shift = 7;
  n = 0;
  x_kw = x + kw;
  y_kh = y + kh;
  for (i = x; i < x_kw; ++i) {
    for (j = y; j < y_kh; ++j) {
      if (i < 0 || i > h-1 || j < 0 || j > w-1) {
        continue;
      }

      bitset = nthbitset_arr(src, idx_4d(zi, zj, i, j, d, w, h));
      dst[idx/8] |= bitset << shift;
      mask = rotr1(mask);
      idx++;
      shift--;
      shift = shift < 0 ? 7 : shift;
      n++;
    }
  }

  return n;
}

float l_conv_pool_bn_bst0_bconv_b[10] = {0.0065972069,-0.0032083243,0.00061612093,0.0019226096,-0.0017260083,-0.0017245957,-0.0020184768,0.00016419064,0.0041997521,0.00042016737};
uint8_t l_conv_pool_bn_bst0_bconv_W[40] = {1,185,241,255,203,221,65,255,176,151,255,255,219,74,26,127,167,156,204,127,176,194,140,127,11,116,223,255,57,223,100,127,33,121,214,255,45,244,0,127};
float l_conv_pool_bn_bst0_bn_beta[10] = {-0.29155785,-0.31650674,-0.1901871,-0.30414388,-0.20494059,-0.10322174,-0.055761814,-0.12312096,-0.25513476,-0.01941151};
float l_conv_pool_bn_bst0_bn_gamma[10] = {0.83450437,0.83689398,1.0502133,0.8664788,1.0448612,0.8839711,1.0796871,1.0928549,0.97461534,0.87475497};
float l_conv_pool_bn_bst0_bn_mean[10] = {1.3481066,1.7951958,3.7045608,0.68163759,1.7699362,-0.17213583,2.4068148,2.3783443,1.6148722,-0.0227382};
float l_conv_pool_bn_bst0_bn_std[10] = {1.8100452,1.9019715,3.5531201,1.2475958,2.2687366,1.3270679,2.3854992,2.8835018,1.8160192,1.578292};
void l_conv_pool_bn_bst0(float* input, uint8_t* output){
  fconv_layer(input, l_conv_pool_bn_bst0_bconv_W, output, l_conv_pool_bn_bst0_bconv_b, l_conv_pool_bn_bst0_bn_gamma, l_conv_pool_bn_bst0_bn_beta, l_conv_pool_bn_bst0_bn_mean, l_conv_pool_bn_bst0_bn_std, 1, 10, 28, 28, 1, 5, 5, 1, 1, 0, 0, 3, 3, 2, 2, 0, 0);
}

float l_b_conv_pool_bn_bst1_bconv_b[10] = {0.00033188047,-1.7023092e-05,0.00044408877,-0.00074976275,-0.00097692595,1.6071281e-06,0.00053015165,-0.00050257129,-0.00048802749,0.00041597427};
uint8_t l_b_conv_pool_bn_bst1_bconv_W[400] = {51,33,64,127,192,24,102,127,51,141,110,127,160,1,208,255,184,198,255,127,156,102,17,255,0,148,98,127,3,136,71,127,9,196,214,127,196,119,17,127,226,129,239,255,199,93,143,127,201,125,254,127,194,48,14,127,231,225,15,255,211,42,52,255,252,29,221,255,199,45,14,127,239,191,191,255,70,52,79,255,32,102,0,127,34,175,157,127,186,255,32,127,66,251,253,127,108,239,224,127,14,145,142,127,79,152,11,127,115,183,240,127,157,126,0,127,32,31,221,127,139,193,207,255,151,62,225,127,85,128,7,127,14,124,240,127,4,14,0,127,100,184,224,127,144,160,206,127,152,166,64,127,243,192,14,255,22,62,240,255,173,164,0,255,9,214,48,255,90,70,57,255,98,148,112,255,139,49,145,127,166,57,66,127,216,98,119,255,56,215,49,255,123,242,51,255,163,137,141,127,6,126,185,255,66,35,61,255,0,55,125,127,224,6,0,255,32,59,251,255,135,14,16,127,71,223,250,255,64,38,249,255,5,127,250,255,248,153,1,127,118,191,250,127,25,192,15,127,48,30,232,127,127,254,135,127,16,0,11,127,67,232,247,127,36,30,251,127,184,1,10,127,72,27,232,127,255,210,125,255,133,223,221,127,95,127,246,255,159,254,49,127,121,127,252,127,149,63,249,127,60,253,216,127,251,254,47,255,179,95,250,127,215,255,243,255,120,159,243,255,6,223,241,255,12,92,179,255,5,190,208,127,57,211,191,255,0,48,151,255,98,223,254,255,67,254,80,127,156,41,203,127,7,187,112,127,58,0,255,255,68,34,86,127,248,24,0,255,148,41,123,255,236,61,0,127,58,184,32,255,5,60,64,127,135,61,219,255,240,25,4,255,48,16,221,255,14,149,162,127};
float l_b_conv_pool_bn_bst1_bn_beta[10] = {-0.24039693,-0.17782551,-0.1589801,-0.17254464,-0.083694607,-0.15440904,-0.3804417,-0.027677715,-0.068359479,-0.017499572};
float l_b_conv_pool_bn_bst1_bn_gamma[10] = {0.93595439,1.0655004,0.96249527,0.92392516,0.97569513,1.0646091,0.87100041,1.1735951,1.0010576,0.79790312};
float l_b_conv_pool_bn_bst1_bn_mean[10] = {42.21096,29.61132,30.284887,27.793257,38.783981,35.208626,22.376442,16.49193,30.93717,30.605955};
float l_b_conv_pool_bn_bst1_bn_std[10] = {24.433701,27.669468,24.143717,16.957659,24.908976,26.263016,19.340561,35.813202,25.481955,17.737335};
void l_b_conv_pool_bn_bst1(uint8_t* input, uint8_t* output){
  bconv_layer(input, l_b_conv_pool_bn_bst1_bconv_W, output, l_b_conv_pool_bn_bst1_bconv_b, l_b_conv_pool_bn_bst1_bn_gamma, l_b_conv_pool_bn_bst1_bn_beta, l_b_conv_pool_bn_bst1_bn_mean, l_b_conv_pool_bn_bst1_bn_std, 1, 10, 11, 11, 10, 5, 5, 1, 1, 2, 2, 3, 3, 2, 2, 0, 0);
}

float l_b_conv_bn_bst2_bconv_b[10] = {0.003842392,0.0048690834,0.00019391054,-0.0037753587,0.006329074,0.0020145117,-0.0004320993,-0.00045060052,0.0065324348,-0.00097343337};
uint8_t l_b_conv_bn_bst2_bconv_W[200] = {251,255,32,127,179,255,196,127,72,127,237,127,226,127,8,255,224,127,242,127,95,127,227,255,60,255,106,127,180,255,143,255,245,255,253,127,236,127,139,255,62,255,98,127,239,255,237,127,180,255,34,127,217,255,64,127,243,127,63,127,255,255,48,127,31,255,7,255,200,255,248,255,228,255,0,127,108,255,236,127,124,127,212,255,183,127,70,255,34,127,255,255,116,255,175,255,31,255,253,127,226,255,113,255,51,127,12,127,83,127,240,127,0,127,255,255,13,127,66,127,229,255,237,127,251,127,243,255,64,255,222,255,119,127,95,255,255,255,255,255,159,255,147,127,122,255,43,127,103,255,81,255,74,127,122,127,104,127,79,255,72,127,3,127,101,127,32,255,64,127,137,127,41,255,49,255,99,255,216,255,123,255,231,255,159,127,148,127,45,255,229,127,146,127,16,127,160,127,191,127};
float l_b_conv_bn_bst2_bn_beta[10] = {-0.033746954,0.12881586,0.023163585,-0.029158551,-0.14406976,-0.22848415,-0.071209289,0.25192705,-0.027856885,0.19545184};
float l_b_conv_bn_bst2_bn_gamma[10] = {0.99705261,0.92897332,0.97413516,0.89310849,0.85837597,0.9079085,0.89605707,0.80080116,0.97401631,1.0916377};
float l_b_conv_bn_bst2_bn_mean[10] = {0.56849831,-3.2391577,-2.9478681,-1.3177375,-0.94303405,5.1734958,-3.6909163,-1.4165262,2.0354371,0.78876138};
float l_b_conv_bn_bst2_bn_std[10] = {10.078199,9.886241,10.992603,12.086958,12.376275,10.936541,12.771812,9.4950609,9.2521296,11.623242};
void l_b_conv_bn_bst2(uint8_t* input, uint8_t* output){
  bconv_layer(input, l_b_conv_bn_bst2_bconv_W, output, l_b_conv_bn_bst2_bconv_b, l_b_conv_bn_bst2_bn_gamma, l_b_conv_bn_bst2_bn_beta, l_b_conv_bn_bst2_bn_mean, l_b_conv_bn_bst2_bn_std, 1, 10, 5, 5, 10, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0);
}

float l_b_conv_bn_bst3_bconv_b[10] = {0.00057456683,0.0011795372,0.0031323878,-0.0027395142,0.001170192,-6.2380786e-05,-0.00023922852,-0.0014816358,0.0023191962,0.001540872};
uint8_t l_b_conv_bn_bst3_bconv_W[200] = {151,255,39,255,34,255,204,255,252,127,184,255,60,127,175,255,222,127,1,255,190,127,236,127,228,127,7,255,51,255,215,255,0,255,130,255,93,255,216,127,230,127,226,255,132,127,40,255,125,127,207,255,11,255,200,127,253,255,224,127,13,255,222,127,188,255,193,127,23,255,96,255,255,127,95,127,147,127,8,255,126,127,147,255,32,127,223,255,255,255,191,255,191,255,111,255,168,127,251,127,255,255,36,255,147,255,247,255,236,127,95,255,84,127,135,127,127,255,154,127,117,127,130,127,9,127,44,127,32,127,175,255,224,127,23,255,19,255,118,255,243,255,139,255,135,255,127,255,92,127,10,127,218,127,31,255,200,127,47,255,31,255,250,255,250,255,7,255,252,255,3,255,255,127,161,255,44,127,107,127,104,255,160,127,253,127,245,255,159,255,0,255,186,255,252,127,236,255,119,127};
float l_b_conv_bn_bst3_bn_beta[10] = {-0.23659179,-0.12705174,0.14389457,-0.013425669,-0.12861934,0.21019498,0.15798853,0.026803004,-0.11822093,-0.098274469};
float l_b_conv_bn_bst3_bn_gamma[10] = {0.98601729,1.0159637,0.86769849,0.87564015,0.79965621,1.0114999,0.76067889,0.76039338,0.67057073,0.99981856};
float l_b_conv_bn_bst3_bn_mean[10] = {0.34972954,-1.6528322,-2.3825433,0.41575536,-0.7898432,-0.9947148,1.1774553,1.5792023,-0.1904787,0.26385298};
float l_b_conv_bn_bst3_bn_std[10] = {12.196125,13.06022,12.493414,12.697702,13.847657,13.700094,11.420638,11.158777,12.717113,12.519019};
void l_b_conv_bn_bst3(uint8_t* input, uint8_t* output){
  bconv_layer(input, l_b_conv_bn_bst3_bconv_W, output, l_b_conv_bn_bst3_bconv_b, l_b_conv_bn_bst3_bn_gamma, l_b_conv_bn_bst3_bn_beta, l_b_conv_bn_bst3_bn_mean, l_b_conv_bn_bst3_bn_std, 1, 10, 5, 5, 10, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0);
}

float l_b_conv_pool_bn_bst4_bconv_b[10] = {0.0012259671,0.0048762541,0.0018752057,0.0010211096,-0.00044839684,0.0030499983,-0.0026141889,-0.003214051,0.0065996298,0.0014805784};
uint8_t l_b_conv_pool_bn_bst4_bconv_W[200] = {91,255,8,127,60,127,222,255,81,255,1,255,32,127,10,127,31,255,13,255,96,127,23,255,103,255,147,127,36,127,0,127,237,255,0,127,4,127,32,127,4,255,252,127,104,255,0,127,236,255,233,255,154,127,82,255,136,127,253,255,13,255,226,127,100,127,129,255,40,127,127,255,45,127,191,127,242,127,227,255,13,255,127,127,251,255,40,127,0,127,223,127,155,255,0,127,228,127,234,127,141,255,182,127,231,127,145,255,254,255,95,255,18,255,215,127,253,127,224,127,252,127,195,127,16,127,227,127,252,127,87,255,255,255,220,127,180,127,34,127,30,255,101,127,237,255,83,255,249,255,255,255,9,255,128,255,26,127,224,127,227,255,219,255,15,255,44,255,183,127,178,127,110,255,209,127,136,127,16,127,13,255,254,127,31,255,252,255,127,255,155,127,2,255,1,255,185,127,112,255};
float l_b_conv_pool_bn_bst4_bn_beta[10] = {0.13157713,-0.27466252,-0.21164462,-0.13679707,-0.15686063,-0.0001954789,-0.069218405,0.081576794,-0.0551049,-0.10729235};
float l_b_conv_pool_bn_bst4_bn_gamma[10] = {1.0441055,1.0105666,0.9950937,0.90848649,0.70741922,1.0501543,1.0912603,1.0037063,0.94500911,1.2509612};
float l_b_conv_pool_bn_bst4_bn_mean[10] = {13.1789,18.079834,16.852896,18.644482,19.125065,17.211267,19.592386,16.303457,15.016413,16.913143};
float l_b_conv_pool_bn_bst4_bn_std[10] = {13.230009,14.159966,11.919355,10.91739,12.644785,12.908494,12.109187,11.807796,9.4707041,14.001931};
void l_b_conv_pool_bn_bst4(uint8_t* input, uint8_t* output){
  bconv_layer(input, l_b_conv_pool_bn_bst4_bconv_W, output, l_b_conv_pool_bn_bst4_bconv_b, l_b_conv_pool_bn_bst4_bn_gamma, l_b_conv_pool_bn_bst4_bn_beta, l_b_conv_pool_bn_bst4_bn_mean, l_b_conv_pool_bn_bst4_bn_std, 1, 10, 5, 5, 10, 3, 3, 1, 1, 1, 1, 3, 3, 2, 2, 0, 0);
}

float l_b_linear_bn_softmax5_bl_b[10] = {-0.00099401537,0.0059561711,0.0044097644,-0.00037424432,0.0062829806,-0.00031027486,-0.0017258576,0.0040907697,0.00040492474,-0.0017686434};
uint8_t l_b_linear_bn_softmax5_bl_W[50] = {20,36,14,199,243,4,255,112,1,17,252,19,51,147,55,255,64,0,4,162,187,253,11,191,60,112,140,168,10,204,64,240,255,53,111,192,5,61,244,1,15,5,248,202,252,115,202,12,248,44};
float l_b_linear_bn_softmax5_bn_beta[10] = {-0.073550232,-0.11921254,0.056880508,0.095163368,-0.023928888,0.14916368,-0.1681412,-0.093775727,0.017108165,0.1054799};
float l_b_linear_bn_softmax5_bn_gamma[10] = {1.7829785,1.8604977,1.7040818,1.7002473,1.7080711,1.5807922,1.7066518,1.8174067,1.6580067,1.6583292};
float l_b_linear_bn_softmax5_bn_mean[10] = {0.29012012,-2.4706893,-1.1905229,0.84207797,-0.20730793,4.385972,-0.68416059,0.38037348,-0.91151702,4.739347};
float l_b_linear_bn_softmax5_bn_std[10] = {8.6553631,12.27835,9.112196,12.192903,9.1133156,8.1899948,10.804722,9.2026663,9.7554817,10.699675};
void l_b_linear_bn_softmax5(uint8_t* input, uint8_t* output){
  blinear_sm_layer(input, l_b_linear_bn_softmax5_bl_W, output, l_b_linear_bn_softmax5_bl_b, l_b_linear_bn_softmax5_bn_gamma, l_b_linear_bn_softmax5_bn_beta, l_b_linear_bn_softmax5_bn_mean, l_b_linear_bn_softmax5_bn_std, 1, 40, 10); 
}


uint8_t temp1[72] = {0};
uint8_t temp2[72] = {0};



void ebnn_compute(float *input, uint8_t *output){
   //perfcounter_config(COUNT_CYCLES, true);

  l_conv_pool_bn_bst0(input, temp1);
  // perfcounter_t run_time = perfcounter_get();
 //  printf("Time: %lu\n", run_time);  

  l_b_conv_pool_bn_bst1(temp1, temp2);
  l_b_conv_bn_bst2(temp2, temp1);
  l_b_conv_bn_bst3(temp1, temp2);
  l_b_conv_pool_bn_bst4(temp2, temp1);
  l_b_linear_bn_softmax5(temp1, output);
}
/*
void ebnn_compute(float *input, uint8_t *output){

  l_conv_pool_bn_bst0(input, temp1);
for(int i=0; i<72; i++){
        printf("%d ", temp1[i]);
}
printf("\n");

  l_b_conv_pool_bn_bst1(temp1, temp2);
for(int i=0; i<72; i++){ 
        printf("%d ", temp2[i]);
}
printf("\n");
 
l_b_conv_bn_bst2(temp2, temp1);
for(int i=0; i<72; i++){ 
        printf("%d ", temp1[i]);
}
printf("\n");
l_b_conv_bn_bst3(temp1, temp2);
for(int i=0; i<72; i++){ 
        printf("%d ", temp2[i]);
}
printf("\n");

l_b_conv_pool_bn_bst4(temp2, temp1);
for(int i=0; i<72; i++){ 
        printf("%d ", temp1[i]);
}
printf("\n");


l_b_linear_bn_softmax5(temp1, output);
for(int i=0; i<72; i++){ 
        printf("%d ", output[i]);
}
printf("\n");
}
*/
int main()
{
float count = 0;
   uint8_t output[1];
   total = 0;
   float* f = mem_alloc(DIGIT_SIZE);

   int* test_label = mem_alloc(156*sizeof(int));
   mram_read(labeldata, test_label, LBUFFER_SIZE);

//uint32_t i = 6;
for(int i = 0; i<156; i++){
   mram_read(inputdata+i*DIGIT_SIZE, f, MAX_READ);
   mram_read(inputdata+i*DIGIT_SIZE+MAX_READ, (void*)f+MAX_READ, DIGIT_SIZE-MAX_READ);
  
   ebnn_compute(f, output);

    int fail =  test_label[i] != (int)output[0];
    if(fail == 0){
	count++;
         total++;
    }else{

    }

    printf("actual: %d %s predicted: %d\n", (int)test_label[i], (fail ? "<>" : "=="), output[0]);
  }

  //printf("count %f\n", count);
  //printf("Correct rate: %f\n", count/156.0);
  return 0;
}
