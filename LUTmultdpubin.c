
#include <stdio.h>
#include <stdint.h>
#include <alloc.h>
#include <mram.h>
#include <perfcounter.h>
#include <limits.h>
#include <float.h>

#define BUFFER_SIZE 15288
#define LBUFFER_SIZE 624
#define DIGIT_SIZE 98
#define MAX_READ 2048



__mram_noinit uint8_t inputdata[BUFFER_SIZE];
__mram_noinit uint8_t labeldata[LBUFFER_SIZE];
__mram_noinit uint8_t LUT[14128];
__host uint32_t total;

__dma_aligned struct look_up_table{
  int16_t bconv_max_m_LUT[1];
  int16_t n_LUT[363];
  int8_t bn_LUT[2560];
  float sm_bn_LUT[2710];
} l;



#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)
#define MAX_FILTER_BYTES 12
const uint8_t bits[8] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};

/* layer types */
void blinear_sm_layer(const uint8_t* A, const uint8_t* F, uint8_t* C,
                             const int16_t m, const int16_t n,
                             const int16_t k);

void bconv_layer(const uint8_t* A, const uint8_t* F, uint8_t* C,const int16_t m,
                        const int16_t num_f, const int16_t w, const int16_t h, const int16_t d,
                        const int16_t kw, const int16_t kh, const int16_t sw, const int16_t sh,
                        const int16_t pw, const int16_t ph, const int16_t pl_w,
                        const int16_t pl_h, const int16_t pl_sw, const int16_t pl_sh,
                        const int16_t pl_pw, const int16_t pl_ph);

/* layer helper functions */
 void bconv(const uint8_t* A, const uint8_t* F, uint8_t* C,
                  const int16_t c_start_idx, const int16_t z, const uint16_t nf, const int16_t w, const int16_t h, const int16_t d,
                  const int16_t kw, const int16_t kh, const int16_t sw, const int16_t sh,
                  const int16_t pw, const int16_t ph, const int16_t pl_w, const int16_t pl_h,
                  const int16_t pl_sw, const int16_t pl_sh, const int16_t pl_pw,
                  const int16_t pl_ph);
 int16_t bdot_3d(const uint8_t* A, const uint8_t* B, const int16_t x, const int16_t y,
                   const int16_t z, const int16_t w, const int16_t h, const int16_t d,
                   const int16_t kw, const int16_t kh);
 int16_t bdot(const uint8_t* A, const uint8_t* B, const int16_t N);

/* indexing functions */
 int16_t idx_2d(const int16_t i, const int16_t j, const int16_t rows);

 int16_t idx_4d(const int16_t i, const int16_t j, const int16_t k, const int16_t l,
                  const int16_t rows, const int16_t cols, const int16_t depth);
 int16_t conv_idx(const int16_t pl_i, const int16_t x, const int16_t kx, const int16_t sx,
                    const int16_t px);
 int16_t convpool_size(const int16_t x, const int16_t kx, const int16_t sx, const int16_t px,
                         const int16_t pl_x, const int16_t pl_sx, const int16_t pl_px);

/* Bit functions */
 uint8_t rotr1 (const uint8_t x);
 int16_t popcnt8(const uint8_t v);
 int16_t nthbitset_arr(const uint8_t* const arr, const int16_t n);
 int16_t bslice_2d_filter(uint8_t* const dst, const uint8_t* const src,
                            const int16_t x, const int16_t y, const int16_t w, const int16_t h,
                            const int16_t kw, const int16_t kh);
 int16_t bslice_4d(uint8_t* const dst, const uint8_t* const src, const int16_t x,
                     const int16_t y, const int16_t zi, const int16_t zj, const int16_t w,
                     const int16_t h, const int16_t d, const int16_t kw, const int16_t kh);

/* layers types */



void blinear_sm_layer(const uint8_t* A, const uint8_t* F, uint8_t* C,
                             const int16_t m, const int16_t n,
                             const int16_t k)
{
  int16_t i, j, ni, ci,  max_idx;
  const uint8_t *Ari; 
  int16_t res;
  float f_res, max_res;

  /* Compute ceil in terms of 8-bits strides */
  ni = (n + 7) / 8;
  for (i = 0; i < m; ++i) {
    max_res = -FLT_MAX;
    Ari = A + (i * ni);
    for (j = 0; j < k; ++j) {
      ci = j * ni;
      res = bdot(Ari, F + ci, n);

      f_res = l.sm_bn_LUT[(res+128)*10+j];


      if (f_res > max_res) {
       max_idx = j;
        max_res = f_res;
      }
    }
    C[i] = max_idx;
  }
}

void bconv_layer(const uint8_t* A, const uint8_t* F, uint8_t* C, const int16_t m,
                        const int16_t num_f, const int16_t w, const int16_t h, const int16_t d,
                        const int16_t kw, const int16_t kh, const int16_t sw, const int16_t sh,
                        const int16_t pw, const int16_t ph, const int16_t pl_w,
                        const int16_t pl_h, const int16_t pl_sw, const int16_t pl_sh,
                        const int16_t pl_pw, const int16_t pl_ph)
{
  int16_t i, j, max_m, res_size, res_w, res_h, c_idx, f_idx, dcp_kwkh;


  c_idx = 0;
  res_w = convpool_size(w, kw, sw, pw, pl_w, pl_sw, pl_pw);
  res_h = convpool_size(h, kh, sh, ph, pl_h, pl_sh, pl_ph);
  res_size = res_w * res_h;

  //360/8.0
  max_m = l.bconv_max_m_LUT[0];
// max_m = CEIL_POS(res_size*m*num_f/8.0); 
//dcp_kwkh = d*CEIL_POS(kw*kh/8.0);
  dcp_kwkh = d*l.n_LUT[kw*kh];

  /* initialize result */
  for (i = 0; i < max_m; ++i) {
   C[i] = 0;
  }

  for (i = 0; i < m; ++i) {
    for (j = 0; j < num_f; ++j) {
      f_idx = j*dcp_kwkh;
      bconv(A, F + f_idx, C, c_idx, i, j, w, h, d, kw, kh, sw, sh, pw, ph,
            pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);
      c_idx += res_size;
    }
  }
}


int16_t bdot(const uint8_t* A, const uint8_t* B, const int16_t N)
{
  int16_t i, num_bytes, res;

  //N = 9//num_bytes = CEIL_POS(N/8.0);
  num_bytes = l.n_LUT[N];
  res = 0;
  for (i = 0; i < num_bytes; ++i) {
    res += popcnt8(~(A[i]^B[i]));
  }
  res = res*2 - N;
  return res;
}

int16_t bdot_3d(const uint8_t* A, const uint8_t* B, const int16_t x, const int16_t y,
                    const int16_t z, const int16_t w, const int16_t h, const int16_t d,
                    const int16_t kw, const int16_t kh)
{
  /* Handles up to 10x10 filters */
  uint8_t A_slice[MAX_FILTER_BYTES] = {0};
  uint8_t B_slice[MAX_FILTER_BYTES] = {0};
  const uint8_t *B_idx;
  int16_t i, comp_n, res, N, B_bytes, bx, by, bw, bh, w_x, h_y;

  N = kw*kh;

  //kw*kh = 9
  B_bytes = l.n_LUT[kw*kh];
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

void bconv(const uint8_t* A, const uint8_t* F, uint8_t* C,
                  const int16_t c_start_idx, const int16_t z, const uint16_t nf, const int16_t w, const int16_t h, const int16_t d,
                  const int16_t kw, const int16_t kh, const int16_t sw, const int16_t sh,
                  const int16_t pw, const int16_t ph, const int16_t pl_w, const int16_t pl_h,
                  const int16_t pl_sw, const int16_t pl_sh, const int16_t pl_pw,
                  const int16_t pl_ph)
{
  uint8_t c_mask, res_sign;
  int16_t pl_i, pl_j, i, j, i_in, j_in, pl_i_max, pl_j_max, c_shift, c_idx, pl_w2_1, pl_hpw_1;
  int8_t res, max_res; //
  //float max_res;

  c_shift = 7 - (c_start_idx % 8);
  c_mask = 1 << c_shift;
  c_idx = c_start_idx / 8;
  pl_i_max = (w - kw + 2*pw)/sw + (2*pl_pw) + 1;
  pl_j_max = (h - kh + 2*ph)/sh + (2*pl_ph) + 1;
  pl_w2_1 = pl_w + pl_pw - 1;
  pl_hpw_1 = pl_h + pl_pw - 1;
  for (pl_i = -pl_pw; pl_i + pl_w2_1 < pl_i_max; pl_i += pl_sw) {
  for (pl_j = -pl_ph; pl_j + pl_hpw_1 < pl_j_max; pl_j += pl_sh) {
    max_res = -128; //////
    int16_t pl_i_pl_w = pl_i + pl_w;
    for (i_in = pl_i; i_in < pl_i_pl_w; ++i_in) {
    i = conv_idx(i_in, w, kw, sw, pw);
    int16_t pl_j_pl_h = pl_j + pl_h;
    for (j_in = pl_j; j_in < pl_j_pl_h; ++j_in) {
      j = conv_idx(j_in, h, kh, sh, ph);
      if (i >= -pw && j >= -ph) {
        res = bdot_3d(A, F, i, j, z, w, h, d, kw, kh);
        max_res = MAX(res, max_res);
      }
    }
    }

    res_sign = l.bn_LUT[(max_res+128)*10+nf];


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

/* index functions */
int16_t idx_2d(const int16_t i, const int16_t j, const int16_t rows)
{
  return i * rows + j;
}

int16_t idx_3d(const int16_t i, const int16_t j, const int16_t k, const int16_t rows,
                  const int16_t cols)
{
  return i * rows * cols + j * cols + k;
}

int16_t idx_4d(const int16_t i, const int16_t j, const int16_t k, const int16_t l,
                  const int16_t rows, const int16_t cols, const int16_t depth)
{
  return i * rows * cols * depth + j * cols * depth + k * depth + l;
}

int16_t conv_idx(const int16_t pl_i, const int16_t x, const int16_t kx, const int16_t sx,
                    const int16_t px)
{
  int16_t conv_sz = (x - kx + 2*px)/sx;
  return (pl_i < 0 || pl_i > conv_sz) ? -INT16_MAX : pl_i * sx - px;
}

int16_t convpool_size(const int16_t x, const int16_t kx, const int16_t sx, const int16_t px,
                         const int16_t pl_x, const int16_t pl_sx, const int16_t pl_px)
{
  return ((x - kx + 2*px)/sx - pl_x + (2*pl_px) + 1)/pl_sx + 1;
}

/* Bit functions */
uint8_t rotr1 (const uint8_t x)
{
  return (x >> 1) | (x << 7);
}

int16_t nthbitset_arr(const uint8_t* const arr, const int16_t n)
{
  return arr[n/8] & bits[n%8] ? 1 : 0;
}

int16_t popcnt8(const uint8_t v) {
  uint8_t c;
  c = v - ((v >> 1) & 0x55);
  c = ((c >> 2) & 0x33) + (c & 0x33);
  return ((c >> 4) + c) & 0x0F;
}


int16_t bslice_2d_filter(uint8_t* const dst, const uint8_t* const src,
                            const int16_t x, const int16_t y, const int16_t w, const int16_t h,
                            const int16_t kw, const int16_t kh)
{
  int16_t i, j, n, idx, shift, bytes, bitset, x_kw, y_kh;
  uint8_t mask;

  /* initiaize dst */
  //9
  bytes = l.n_LUT[kw*kh];
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
int16_t bslice_4d(uint8_t* const dst, const uint8_t* const src, const int16_t x,
                     const int16_t y, const int16_t zi, const int16_t zj, const int16_t w,
                     const int16_t h, const int16_t d, const int16_t kw, const int16_t kh)
{
  int16_t i, j, n, idx, shift, bytes, bitset, x_kw, y_kh;
  uint8_t mask;

  /* initialize dest */
  //9
  //bytes = CEIL_POS(kw*kh/8.0);
  bytes = l.n_LUT[kw*kh];
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
uint8_t l_b_conv_pool_bn_bst0_bconv_W[20] = {248,127,228,127,228,127,110,255,189,255,158,255,183,127,19,255,91,127,101,255};
void l_b_conv_pool_bn_bst0(uint8_t* input, uint8_t* output){
  bconv_layer(input, l_b_conv_pool_bn_bst0_bconv_W, output, 1, 10, 28, 28, 1, 3, 3, 2, 2, 0, 0, 3, 3, 2, 2, 0, 0);
}

uint8_t l_b_linear_bn_softmax1_bl_W[450] = {98,3,128,79,224,36,56,24,94,89,65,130,3,146,36,111,53,216,250,126,179,43,19,207,196,146,185,33,118,121,236,203,131,192,67,7,98,118,57,60,37,46,207,61,245,0,12,161,37,160,32,75,24,96,34,2,34,129,10,208,32,82,19,188,75,41,32,59,197,2,152,115,167,68,129,21,37,192,164,28,167,128,5,49,85,176,128,180,18,106,113,238,177,255,31,60,240,59,243,244,239,1,15,126,63,153,222,32,239,225,53,159,55,127,3,201,211,56,240,116,253,155,18,211,240,195,91,104,140,20,18,59,29,78,113,35,204,204,219,226,60,172,205,254,59,130,30,243,0,38,50,137,248,230,10,142,203,199,176,190,238,253,163,59,167,143,206,236,242,191,56,255,40,243,202,100,36,56,169,244,2,16,111,248,88,17,2,189,129,63,0,59,236,113,247,95,78,46,125,61,178,80,30,235,235,176,3,102,18,146,82,15,208,50,116,115,255,135,203,129,208,223,89,97,15,8,115,252,193,248,199,183,196,31,192,119,254,107,137,9,65,9,197,78,119,124,201,214,231,83,78,219,1,2,5,13,114,79,30,127,12,100,147,32,66,246,113,192,9,227,188,113,3,124,235,247,146,124,222,134,48,102,99,235,236,53,246,192,20,140,119,88,233,200,149,98,168,126,7,71,62,0,40,124,96,32,118,143,222,159,15,128,213,182,192,255,63,242,76,3,241,113,36,160,71,159,35,136,3,32,146,103,25,1,228,162,27,144,63,90,33,53,38,173,51,51,36,31,195,120,48,3,56,243,61,194,50,28,103,120,240,124,199,34,250,12,237,243,15,140,102,75,25,195,53,157,134,143,63,219,197,160,227,223,61,226,15,141,251,206,104,115,175,19,56,134,57,243,206,112,115,127,127,142,127,230,227,29,114,52,94,51,222,129,10,240,152,96,28,8,251,73,28,129,252,184,48,28,15,29,230,97,192,239,255,199,211,191,116,138,59,0,238,226,14,217,152,207,243,103};
void l_b_linear_bn_softmax1(uint8_t* input, uint8_t* output){
  blinear_sm_layer(input, l_b_linear_bn_softmax1_bl_W, output, 1, 360, 10); 
}

uint8_t temp1[104] = {0};
uint8_t temp2[104] = {0};

void ebnn_compute(uint8_t *input, uint8_t *output){

  l_b_conv_pool_bn_bst0(input, temp1);

  l_b_linear_bn_softmax1(temp1, output);
}


int main()
{
 

  uint8_t output[1];

   uint8_t* input = mem_alloc(12*DIGIT_SIZE);

   int* test_label = mem_alloc(156*sizeof(int));
   mram_read(labeldata, test_label, LBUFFER_SIZE);

   for(int i = 0; i < 6; i++){

   	mram_read(LUT+2048*i, ((void*)&l)+2048*i, 2048);
   }

   mram_read(LUT+2048*6, ((void*)&l)+2048*6,14128-(2048*6));

  printf("%d\n", l.bconv_max_m_LUT[0]);
 for(int16_t i= 0; i < 1; i++) {

   mram_read(inputdata+i*12*DIGIT_SIZE, input, 12*DIGIT_SIZE);
   for(int16_t j = 0; j < 1; j++){
   	ebnn_compute(&input[98*j], output);
   
 
    int16_t fail =  test_label[i*12+j] != (int)output[0];
    if(fail == 0){
         total++;
    }else{

    }

//    printf("actual: %d %s predicted: %d\n", (int)test_label[i*12+j], (fail ? "<>" : "=="), output[0]);
    }
   }
  
  return 0;
}


