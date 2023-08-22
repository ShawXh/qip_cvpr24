#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <queue>
#include <string.h>
#include <vector>
#include <random>
#include <complex>
#include<unordered_map>

using namespace std;

#if defined(__AVX2__) ||                                                       \
    defined(__FMA__) // icpc, gcc and clang register __FMA__, VS does not
#define VECTORIZE 1
#define AVX_LOOP _Pragma("omp simd")
#else
#define AVX_LOOP // empty
#endif

#ifndef UINT64_C // VS can not detect the ##ULL macro
#define UINT64_C(c) (c##ULL)
#endif

#define SIGMOID_BOUND 6.0  // computation range for fast sigmoid lookup table
#define DEFAULT_ALIGN 128  // default align in bytes
#define MAX_CODE_LENGTH 64 // maximum HSM code length. sufficient for nv < int64

std::mt19937 rng(0);
std::uniform_real_distribution<double> randf(0., 1.);

// By Hao
float *exp_table;
// params for RNCE
float beta = 0.;
float gam = 1.; // the parameter of Gaussian/Laplacian kernel
int reg = 0;
/*
 * 0 for no distance function
 * 1 for wasser-1
 * 2 for wasser-2
 * 3 for wasser-3, (* emprically best)
 * 4 for gaussian
 * 5 for laplacian
 */
float ub_w = 0; // the upper bound of lambda_dis * beta, a recommend value: 0.1 for wasser-1 and laplacian
const float EXP_BOUND = 6;
// 

int num_qubits = 6;
int sample_times = 2;
char out_mode[100] = "mode";
int norm2 = 1;
int qip = 0;

using namespace std;

typedef unsigned long long ull;
typedef unsigned int uint;
typedef unsigned char byte;

int verbosity = 2; // verbosity level. 2 = report progress and tell jokes, 1 =
                   // report time and hsm size, 0 = errors, <0 = shut up
int n_threads = 8; // number of threads program will be using
float initial_lr = 0.025f; // initial learning rate
int n_hidden = 128;   // DeepWalk parameter "d" = embedding dimensionality aka
                      // number of nodes in the hidden layer
int n_walks = 40;     // DeepWalk parameter "\gamma" = walks per vertex
int walk_length = 80; // DeepWalk parameter "t" = length of the walk
int window_size = 10; // DeepWalk parameter "w" = window size
int n_neg_samples = 5;
float p = 1;
float q = 1;

ull step = 0; // global atomically incremented step counter

ull nv = 0, ne = 0; // number of nodes and edges
                    // We use CSR format for the graph matrix (unweighted).
                    // Adjacent nodes for vertex i are stored in
                    // edges[offsets[i]:offsets[i+1]]
int *offsets;       // CSR index pointers for nodes.
int *edges;         // CSR offsets
int *degrees;       // Node degrees
int *train_order;   // We shuffle the nodes for better performance
ull *edge_offsets;  // Alias table pointers

int *n2v_js;
float *n2v_qs;

int *neg_js;
float *neg_qs;
float *node_cnts;

float *wVtx; // Vertex embedding, aka DeepWalk's \Phi
float *wCtx; // Context embedding

const int sigmoid_table_size = 1024; // This should fit in L1 cache
const float SIGMOID_RESOLUTION = sigmoid_table_size / (SIGMOID_BOUND * 2.0f);
float *sigmoid_table;

// http://xoroshiro.di.unimi.it/#shootout
// We use xoroshiro128+, the fastest generator available
uint64_t rng_seed[2];

void init_rng(uint64_t seed) {
  for (int i = 0; i < 2; i++) {
    ull z = seed += UINT64_C(0x9E3779B97F4A7C15);
    z = (z ^ z >> 30) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ z >> 27) * UINT64_C(0x94D049BB133111EB);
    rng_seed[i] = z ^ (z >> 31);
  }
}

static inline uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

uint64_t lrand() {
  const uint64_t s0 = rng_seed[0];
  uint64_t s1 = rng_seed[1];
  const uint64_t result = s0 + s1;
  s1 ^= s0;
  rng_seed[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14);
  rng_seed[1] = rotl(s1, 36);
  return result;
}

static inline double drand() {
  const union un {
    uint64_t i;
    double d;
  } a = {UINT64_C(0x3FF) << 52 | lrand() >> 12};
  return a.d - 1.0;
}

inline int irand(uint32_t max) {
  uint32_t rnd = lrand();
  return (uint64_t(rnd) * uint64_t(max)) >> 32;
}

inline int irand(uint32_t min, uint32_t max) { return irand(max - min) + min; }

inline void *
aligned_malloc(size_t size,
               size_t align) { // universal aligned allocator for win & linux
#ifndef _MSC_VER
  void *result;
  if (posix_memalign(&result, align, size))
    result = 0;
#else
  void *result = _aligned_malloc(size, align);
#endif
  return result;
}

inline void aligned_free(void *ptr) { // universal aligned free for win & linux
#ifdef _MSC_VER
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

void init_sigmoid_table() { // this shoould be called before fast_sigmoid once
  sigmoid_table = static_cast<float *>(
      aligned_malloc((sigmoid_table_size + 1) * sizeof(float), DEFAULT_ALIGN));
  for (int k = 0; k != sigmoid_table_size; k++) {
    float x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
    sigmoid_table[k] = 1 / (1 + exp(-x));
  }
}

float fast_sigmoid(float x) {
  if (x > SIGMOID_BOUND)
    return 1;
  if (x < -SIGMOID_BOUND)
    return 0;
  int k = (x + SIGMOID_BOUND) * SIGMOID_RESOLUTION;
  return sigmoid_table[k];
}

void InitExpTable()
{
    float x;
    exp_table = (float *)malloc((sigmoid_table_size) * sizeof(float));
    for (int k = 0; k != sigmoid_table_size; k++)
    {
        x = 2.0 * EXP_BOUND * k / sigmoid_table_size - EXP_BOUND;
        exp_table[k] = exp(x);
    }
}

float FastExp(float x)
{
    if (x > EXP_BOUND) return exp_table[sigmoid_table_size - 1];
    else if (x < -EXP_BOUND) return exp_table[0];
    int k = (x + EXP_BOUND) * sigmoid_table_size / EXP_BOUND / 2;
    return exp_table[k];
}

inline int sample_neighbor(int node) { // sample neighbor node from a graph
  if (offsets[node] == offsets[node + 1])
    return -1;
  return edges[irand(offsets[node], offsets[node + 1])];
}

inline int has_edge(int from, int to) {
  return binary_search(&edges[offsets[from]], &edges[offsets[from + 1]], to);
}

void shuffle(int *a, int n) { // shuffles the array a of size n
  for (int i = n - 1; i >= 0; i--) {
    int j = irand(i + 1);
    int temp = a[j];
    a[j] = a[i];
    a[i] = temp;
  }
}

int ArgPos(char *str, int argc, char **argv) {
  for (int a = 1; a < argc; a++)
    if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        cout << "Argument missing for " << str << endl;
        exit(1);
      }
      return a;
    }
  return -1;
}

// Regularizer
inline void UpdateReg(float *vec_u, float *vec_v, float *vec_error, float sigmoid_score, float lr)
{
	float lam_dis, dis, dis2, g;
	lam_dis = 0;
	dis2 = 0;
	
	if (!reg) return;

	switch(reg) {
		case 1:
			// Wasserstein-1
			for (int c = 0; c != n_hidden; c++) dis2 += (vec_u[c] - vec_v[c]) * (vec_u[c] - vec_v[c]);
			lam_dis = 1. / (sqrt(dis2) + 1e-8);
			break;
		case 2:
			// Wasserstein-2
			lam_dis = 1.; // should be 2 but 1 will not hurt
			break;
		case 3:
			// Wasserstein-3
			for (int c = 0; c != n_hidden; c++) dis2 += (vec_u[c] - vec_v[c]) * (vec_u[c] - vec_v[c]);
			lam_dis = 3 * sqrt(dis2);
			break;
		case 4:
			// Gaussian
			for (int c = 0; c != n_hidden; c++) dis2 += (vec_u[c] - vec_v[c]) * (vec_u[c] - vec_v[c]);
			lam_dis = gam * FastExp(-gam * dis2) * 2;
			break;
		case 5:
			// Laplacian
			for (int c = 0; c != n_hidden; c++) dis2 += (vec_u[c] - vec_v[c]) * (vec_u[c] - vec_v[c]);
			dis = sqrt(dis2);
			lam_dis = gam * FastExp(-gam * dis) / (dis + 1e-8);
			break;
		default:
			break;
	}

	lam_dis *= beta;
	if (ub_w > 1e-6 && lam_dis > ub_w) lam_dis = ub_w;

	for (int c = 0; c != n_hidden; c++) {
	    g = lam_dis * (vec_u[c] - vec_v[c]) * lr;
	    vec_error[c] += -g;
	    vec_v[c] += g;
	}
}

// inline void normalize(float *vec) {
//   float length = 0;
//   AVX_LOOP
//   for (int c = 0; c < n_hidden; c++) length += vec[c] * vec[c];
//   length = sqrt(length);
//   AVX_LOOP
//   for (int c = 0; c < n_hidden; c++) vec[c] /= length;
// }


inline float length(float *vec) {
  float x = 0;
  for (int c = 0; c < n_hidden; c++) x += vec[c] * vec[c];
  x = sqrt(x);
  return x;
}

/* Update embeddings */
inline void my_update(float *vec_u, float *vec_v, float *vec_error, float lr, int label)
{
    float x, g, score;
	  x = 0;
    for (int c = 0; c != n_hidden; c++) x += vec_u[c] * vec_v[c];
    score = fast_sigmoid(x);

    g = (label - score) * lr;
    
    if (vec_u == vec_v) {
        g = g * 2;
        for (int c = 0; c != n_hidden; c++) vec_error[c] += g * vec_v[c];
    } else {
        for (int c = 0; c != n_hidden; c++) vec_error[c] += g * vec_v[c];
		// Regularized NCE
		if (reg && label)
			UpdateReg(vec_u, vec_v, vec_error, score, lr);
        for (int c = 0; c != n_hidden; c++) vec_v[c] += g * vec_u[c];
    }
}

int fast_pow2(int n) {
  return 1 << n;
}

void RecordIdx(unordered_map<int, int> &idx_count, const int idx) {
  if (idx_count.find(idx) == idx_count.end()) idx_count[idx] = 1;
  else (idx_count[idx])++;
}

float ComputeProbHand(int idx, int num_states, float phi) {
  complex<double> tmp;
  tmp = 0. + 0i;
  for (int k = 0; k < num_states; k++)
      tmp += exp(2i * M_PI * (double)k * (-idx / (double)num_states + phi));
  return (tmp * conj(tmp)).real() / ((float)num_states * (float)num_states);
}

float GetProb(
    std::vector<float> &probs, 
    int &idx, 
    int &compute_iter, 
    const int num_states, 
    const float phi) {
  float prob;
  if (probs.size() < compute_iter + 1) {
    prob = ComputeProbHand(idx, num_states, phi);
    // prob = ComputeProbTh(idx, num_states, phi);
    probs.push_back(prob);
  } else {
    prob = probs[compute_iter];
    // cout << "fetch probs[" << compute_iter << "]: " << prob << endl;
  }
  return prob;
}

float FastSampleQPEOnce(int num_qubits, const float gt_ip, bool debug=false) {
  if (debug) {
    cout << num_qubits << " Quibits" << endl;
    cout << "Ground truth: " << gt_ip << endl;
  }
  if (gt_ip > 1.) return 1.;
  if (gt_ip < -1.) return -1.;

  float phi = 0.5 * acos(-gt_ip) / M_PI;
  if (debug) cout << "gt R: " << phi << endl;

  int num_states = fast_pow2(num_qubits);
  if (debug) cout << num_states << " states" << endl;

  int idx = 0;

  // fast sampling
  float p = randf(rng);
  if (debug) cout << "random number p: " << p << endl;

  float p1 = 0., p2 = 0.;
  int idx1 = 0, idx2 = 0;

  idx1 = (int)ceil(num_states * phi) % num_states;
  p1 = ComputeProbHand(idx1, num_states, phi);
  // p1 = ComputeProbTh(idx1, num_states, phi);
  if (debug) cout << "idx1: " << idx1 << ", p1: " << p1 << endl;
  p -= p1;
  if (p < 0) return -cos(2 * M_PI * idx1 / num_states);
  idx2 = (int)floor(num_states * phi);
  p2 = ComputeProbHand(idx2, num_states, phi);
  // p2 = ComputeProbTh(idx2, num_states, phi);
  if (debug) cout << "idx2: " << idx2 << ", p2: " << p2 << endl;
  p -= p2;
  if (p < 0) return -cos(2 * M_PI * idx2 / num_states);

  // other cases
  for (int i = 0; i < num_states / 2 - 1; i++) {
    // idx1
    idx1 = (idx1 + 1) % num_states;
    p1 = ComputeProbHand(idx1, num_states, phi);
    // p1 = ComputeProbTh(idx1, num_states, phi);
    if (debug) cout << "idx1: " << idx1 << ", p1: " << p1 << endl;
    p -= p1;
    if (p < 0) return -cos(2 * M_PI * (float)idx1 / num_states);
    // idx2
    idx2 = ((idx2 - 1) + num_states) % num_states;
    // p2 = ComputeProbTh(idx2, num_states, phi);
    p2 = ComputeProbHand(idx2, num_states, phi);
    if (debug) cout << "idx2: " << idx2 << ", p2: " << p2 << endl;
    p -= p2;
    if (p < 0) return -cos(2 * M_PI * (float)idx2 / num_states);
  }

  cout << "WARNING: QPE have searched over all states" << endl;
  return -cos(2 * M_PI * (float)idx2 / num_states);
}

float FastSampleQPE(int num_qubits, float gt_ip, int sample_times=0, bool debug=false, char *out_mode="mode") {
  
  if (sample_times == 0) return FastSampleQPEOnce(num_qubits, gt_ip, debug);


  sample_times = sample_times * 2 + 1;

  if (debug) {
    cout << "Output mode: " << out_mode << endl;
    cout << num_qubits << " Quibits" << endl;
    cout << "Ground truth: " << gt_ip << endl;
  }
  if (gt_ip > 1.) return 1.;
  if (gt_ip < -1.) return -1.;

  float phi = 0.5 * acos(-gt_ip) / M_PI;
  if (debug) cout << "gt R: " << phi << endl;

  int num_states = fast_pow2(num_qubits);
  if (debug) cout << num_states << " states" << endl;

  // fast sampling
  float p;
  

  float p1 = 0., p2 = 0.;
  int idx1 = 0, idx2 = 0;
  int compute_iter = 0;

  std::unordered_map<int, int> idx_count;
  idx_count.reserve(sample_times * 10);
  std::vector<float> probs;

  for (int iter = 0; iter < sample_times; iter++) {
    p = randf(rng);
    if (debug) cout << "Sample iteration " << iter << ", random number p: " << p << endl;
    compute_iter = 0;

    idx1 = (int)ceil(num_states * phi) % num_states;
    p1 = GetProb(probs, idx1, compute_iter, num_states, phi);
    if (debug) cout << "idx1: " << idx1 << ", p1: " << p1 << endl;
    compute_iter++;
    p -= p1;
    if (p < 0) {
      RecordIdx(idx_count, idx1);
      continue;
    }

    idx2 = (int)floor(num_states * phi);
    p2 = GetProb(probs, idx2, compute_iter, num_states, phi);
    if (debug) cout << "idx2: " << idx2 << ", p2: " << p2 << endl;
    compute_iter++;
    p -= p2;
    if (p < 0) {
      RecordIdx(idx_count, idx2);
      continue;
    }

    // other cases
    for (int i = 0; i < num_states / 2 - 1; i++) {
      // idx1
      idx1 = (idx1 + 1) % num_states;
      p1 = GetProb(probs, idx1, compute_iter, num_states, phi);
      if (debug) cout << "idx1: " << idx1 << ", p1: " << p1 << endl;
      compute_iter++;
      p -= p1;
      if (p < 0) {
        RecordIdx(idx_count, idx1);
        break;
      }

      // idx2
      idx2 = ((idx2 - 1) + num_states) % num_states;
      p2 = GetProb(probs, idx2, compute_iter, num_states, phi);
      if (debug) cout << "idx2: " << idx2 << ", p2: " << p2 << endl;
      compute_iter++;
      p -= p2;
      if (p < 0) {
        RecordIdx(idx_count, idx2);
        break;
      }
    }
  }

  int idx = 0, max_time = 0;
  float sum_score = 0.;
  if (!strcmp(out_mode, "mode")) {
    for (auto i = idx_count.cbegin(); i != idx_count.cend(); i++) {
      if (i->second > max_time) {max_time = i->second; idx = i->first;}
    }
    if (debug) cout << "Final idx: " << idx << " times: " << max_time << endl;
    return -cos(2 * M_PI * (float)idx / num_states);
  } else if (!strcmp(out_mode, "avg")) {
    if (debug) cout << "View idx_count..." << endl;
    for (auto i = idx_count.cbegin(); i != idx_count.cend(); i++) {
      idx = i->first;
      sum_score += -cos(2 * M_PI * (float)idx / num_states) * i->second;
      if (debug) cout << "idx " << idx << " times: " << i->second << endl;
    }
    return sum_score / sample_times;
  } else {
    cout << "Error: out_mode " << out_mode << " is invalid!" << endl;
    exit(1);
  }
}

inline void qip_update( // update the embedding, putting w_t gradient in w_t_cache
    float *w_s, float *w_t, float *w_t_cache, float lr, const int label) {
  float score = 0; // score = dot(w_s, w_t)
  AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    score += w_s[c] * w_t[c];
  float l1 = length(w_s);
  float l2 = length(w_t);
  score /= l1 * l2;
  score = FastSampleQPE(num_qubits, score, sample_times, false, out_mode);
  score *= l1 * l2;
  score = (label - 1 / (1 + exp(-score))) * lr;
  AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    w_t_cache[c] += score * w_s[c]; // w_t gradient
  AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    w_s[c] += score * w_t[c]; // w_s gradient

  // normalize(w_s);
}


inline void update( // update the embedding, putting w_t gradient in w_t_cache
    float *w_s, float *w_t, float *w_t_cache, float lr, const int label) {
  float score = 0; // score = dot(w_s, w_t)
  AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    score += w_s[c] * w_t[c];
  score = (label - fast_sigmoid(score)) * lr;
  AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    w_t_cache[c] += score * w_s[c]; // w_t gradient
  AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    w_s[c] += score * w_t[c]; // w_s gradient

  // if (norm2) normalize(w_s);
}

void init_walker(int n, int *j, float *probs) { // assumes probs are normalized
  vector<int> smaller, larger;
  for (int i = 0; i < n; i++) {
    if (probs[i] < 1)
      smaller.push_back(i);
    else
      larger.push_back(i);
  }
  while (smaller.size() != 0 && larger.size() != 0) {
    int small = smaller.back();
    smaller.pop_back();
    int large = larger.back();
    larger.pop_back();
    j[small] = large;
    probs[large] += probs[small] - 1;
    if (probs[large] < 1)
      smaller.push_back(large);
    else
      larger.push_back(large);
  }
}

int walker_draw(const int n, float *q, int *j) {
  int kk = int(floor(drand() * n));
  return drand() < q[kk] ? kk : j[kk];
}

void Train() {
  ull total_steps = n_walks * nv;
  const float subsample = 1e-3 * nv * n_walks * walk_length;
#pragma omp parallel num_threads(n_threads)
  {
    int tid = omp_get_thread_num();
    const int trnd = irand(nv);
    ull ncount = 0;
    ull local_step = 0;
    float lr = initial_lr;
    int *dw_rw = static_cast<int *>(
        aligned_malloc(walk_length * sizeof(int),
                       DEFAULT_ALIGN)); // we cache one random walk per thread
    float *cache = static_cast<float *>(aligned_malloc(
        n_hidden * sizeof(float),
        DEFAULT_ALIGN)); // cache for updating the gradient of a node
#pragma omp barrier
    while (true) {
      if (ncount > 10) { // update progress every now and then
#pragma omp atomic
        step += ncount;
        if (step > total_steps) // note than we may train for a little longer
                                // than user requested
          break;
        if (tid == 0)
          if (verbosity > 1)
            cout << fixed << setprecision(6) << "\rlr " << lr << ", Progress "
                 << setprecision(2) << step * 100.f / (total_steps + 1) << "%";
        ncount = 0;
        local_step = step;
        lr =
            initial_lr *
            (1 - step / static_cast<float>(total_steps + 1)); // linear LR decay
        if (lr < initial_lr * 0.0001)
          lr = initial_lr * 0.0001;
      }
      dw_rw[0] = train_order[(local_step + ncount + trnd) % nv];
      if (degrees[dw_rw[0]] == 0) {
        ncount++;
        continue;
      }
      ull lastedgeidx = offsets[dw_rw[0]] + irand(offsets[dw_rw[0] + 1] - offsets[dw_rw[0]]);
      dw_rw[1] = edges[lastedgeidx];
      for (int i = 2; i < walk_length; i++) {
        int lastnode = dw_rw[i - 1];
        if (degrees[lastnode] == 0) {
          dw_rw[i] = -2;
          break;
        }
        lastedgeidx =
            offsets[lastnode] + walker_draw(degrees[lastnode],
                                            &n2v_qs[edge_offsets[lastedgeidx]],
                                            &n2v_js[edge_offsets[lastedgeidx]]);
        dw_rw[i] = edges[lastedgeidx];
      }

      for (int dwi = 0; dwi < walk_length; dwi++) {
        int b = irand(window_size); // subsample window size
        if (dw_rw[dwi] < 0)
          break;
	size_t n1 = dw_rw[dwi];
        if ((sqrt(node_cnts[n1] / subsample) + 1) * subsample / node_cnts[n1] <
            drand()) // randomly subsample frequent nodes
          continue;
        for (int dwj = max(0, dwi - window_size + b);
             dwj < min(dwi + window_size - b + 1, walk_length); dwj++) {
          if (dwi == dwj)
            continue;
          if (dw_rw[dwj] < 0)
            break;
	  size_t n2 = dw_rw[dwj];

          memset(cache, 0, n_hidden * sizeof(float)); // clear cache
          //update(&wCtx[n1 * n_hidden], &wVtx[n2 * n_hidden], cache, lr, 1);
          if (qip) qip_update(&wCtx[n1 * n_hidden], &wVtx[n2 * n_hidden], cache, lr, 1);
          else update(&wCtx[n1 * n_hidden], &wVtx[n2 * n_hidden], cache, lr, 1);
          // my_update(&wVtx[n2 * n_hidden], &wCtx[n1 * n_hidden], cache, lr, 1);
          for (int i = 0; i < n_neg_samples; i++) {
            size_t neg = walker_draw(nv, neg_qs, neg_js);
            while (neg == n2)
              neg = walker_draw(nv, neg_qs, neg_js);
            if (qip) qip_update(&wCtx[neg * n_hidden], &wVtx[n2 * n_hidden], cache, lr, 0);
            else update(&wCtx[neg * n_hidden], &wVtx[n2 * n_hidden], cache, lr, 0);
			//update(&wCtx[neg * n_hidden], &wVtx[n2 * n_hidden], cache, lr, 0);
          }
          AVX_LOOP
          for (int c = 0; c < n_hidden; c++)
            wVtx[n2 * n_hidden + c] += cache[c];
          // if (qip || norm2) normalize(&wVtx[n2 * n_hidden]);
        }
      }
      ncount++;
    }
  }
}

void Output(string embedding_file)
{
    printf("Writiing Embedding...\n");
    
    FILE *fo;
	char embf[100];
	strcpy(embf, embedding_file.c_str());

  cout << n_hidden << endl;

    // writing embedding file
    fo = fopen(embf, "wb");
    fprintf(fo, "%lld %d\n", nv, n_hidden);
    for (int a = 0; a < nv; a++) {
        fprintf(fo, "%d ", a);
        for (int b = 0; b < n_hidden; b++) fprintf(fo, "%f ", wVtx[a * n_hidden + b]);                
        fprintf(fo, "\n");
    }
    fclose(fo);
}

int main(int argc, char **argv) {
  int a;
  string network_file, embedding_file;
  ull seed = time(nullptr); // default seed is somewhat random
  init_sigmoid_table();
  if ((a = ArgPos(const_cast<char *>("-input"), argc, argv)) > 0)
    network_file = argv[a + 1];
  else {
    if (verbosity > 0)
      cout << "Input file not given! Aborting now.." << endl;
    return 1;
  }
  if ((a = ArgPos(const_cast<char *>("-output"), argc, argv)) > 0)
    embedding_file = argv[a + 1];
  else {
    if (verbosity > 0)
      cout << "Output file not given! Aborting now.." << endl;
    return 1;
  }
  if ((a = ArgPos(const_cast<char *>("-dim"), argc, argv)) > 0)
    n_hidden = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-reg"), argc, argv)) > 0)
    reg = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-norm"), argc, argv)) > 0)
    norm2 = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-qip"), argc, argv)) > 0)
    qip = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-num-qubits"), argc, argv)) > 0)
    num_qubits = atof(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-sample-times"), argc, argv)) > 0)
    sample_times = atof(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-mode"), argc, argv)) > 0)
    strcpy(out_mode, argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-beta"), argc, argv)) > 0)
    beta = atof(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-gamma"), argc, argv)) > 0)
    gam = atof(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-ub-w"), argc, argv)) > 0)
    ub_w = atof(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-seed"), argc, argv)) > 0)
    seed = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-verbose"), argc, argv)) > 0)
    verbosity = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-threads"), argc, argv)) > 0)
    n_threads = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-lr"), argc, argv)) > 0)
    initial_lr = atof(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-p"), argc, argv)) > 0)
    p = atof(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-q"), argc, argv)) > 0)
    q = atof(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-nwalks"), argc, argv)) > 0)
    n_walks = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-walklen"), argc, argv)) > 0)
    walk_length = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-window"), argc, argv)) > 0)
    window_size = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-nsamples"), argc, argv)) > 0)
    n_neg_samples = atoi(argv[a + 1]);
  init_rng(seed);
  ifstream embFile(network_file, ios::in | ios::binary);
  if (embFile.is_open()) {
    char header[] = "----";
    embFile.seekg(0, ios::beg);
    embFile.read(header, 4);
    if (strcmp(header, "XGFS") != 0) {
      if (verbosity > 0)
        cout << "Invalid header!: " << header << endl;
      return 1;
    }
    embFile.read(reinterpret_cast<char *>(&nv), sizeof(long long));
    embFile.read(reinterpret_cast<char *>(&ne), sizeof(long long));
    offsets = static_cast<int *>(
        aligned_malloc((nv + 1) * sizeof(int32_t), DEFAULT_ALIGN));
    edges =
        static_cast<int *>(aligned_malloc(ne * sizeof(int32_t), DEFAULT_ALIGN));
    embFile.read(reinterpret_cast<char *>(offsets), nv * sizeof(int32_t));
    offsets[nv] = static_cast<int>(ne);
    embFile.read(reinterpret_cast<char *>(edges), sizeof(int32_t) * ne);
    if (verbosity > 0)
      cout << "nv: " << nv << ", ne: " << ne << endl;
    embFile.close();
  } else {
    return 0;
  }
  wVtx = static_cast<float *>(
      aligned_malloc(nv * n_hidden * sizeof(float), DEFAULT_ALIGN));
  for (int i = 0; i < nv * n_hidden; i++)
    wVtx[i] = (drand() - 0.5) / n_hidden;
  wCtx = static_cast<float *>(
      aligned_malloc(nv * n_hidden * sizeof(float), DEFAULT_ALIGN));
  memset(wCtx, 0, nv * n_hidden * sizeof(float));
  train_order =
      static_cast<int *>(aligned_malloc(nv * sizeof(int), DEFAULT_ALIGN));
  for (int i = 0; i < nv; i++)
    train_order[i] = i;
  shuffle(train_order, nv);
  degrees =
      static_cast<int *>(aligned_malloc(nv * sizeof(int32_t), DEFAULT_ALIGN));
  for (int i = 0; i < nv; i++)
    degrees[i] = offsets[i + 1] - offsets[i];
  edge_offsets =
      static_cast<ull *>(aligned_malloc((ne + 1) * sizeof(ull), DEFAULT_ALIGN));
  edge_offsets[0] = 0;
  for (int i = 0; i < ne; i++)
    edge_offsets[i + 1] =
        edge_offsets[i] + offsets[edges[i] + 1] - offsets[edges[i]];

  cout << "Need " << float(edge_offsets[ne]) * 8 / 1024 / 1024
       << " Mb for storing second-order degrees" << endl;
  n2v_qs = static_cast<float *>(
      aligned_malloc(edge_offsets[ne] * sizeof(float), DEFAULT_ALIGN));
  n2v_js = static_cast<int *>(
      aligned_malloc(edge_offsets[ne] * sizeof(int), DEFAULT_ALIGN));
  memset(n2v_js, 0, edge_offsets[ne] * sizeof(float));
#pragma omp parallel for num_threads(n_threads)
  for (int src = 0; src < nv; src++) {
    for (ull dsti = offsets[src]; dsti < offsets[src + 1]; dsti++) {
      int dst = edges[dsti];
      double sum = 0;
      int dst_degree = degrees[dst];
      for (ull dstadji = offsets[dst]; dstadji < offsets[dst + 1]; dstadji++) {
        int dstadj = edges[dstadji];
        ull curidx = edge_offsets[dsti] + dstadji - offsets[dst];
        if (dstadj == src) {
          n2v_qs[curidx] = 1 / p;
          sum += 1 / p;
        } else {
          if (has_edge(dstadj, src)) {
            n2v_qs[curidx] = 1;
            sum += 1;
          } else {
            n2v_qs[curidx] = 1 / q;
            sum += 1 / q;
          }
        }
      }
#pragma omp simd
      for (ull i = edge_offsets[dsti]; i < edge_offsets[dsti] + dst_degree; i++)
        n2v_qs[i] *= dst_degree / sum;
      init_walker(dst_degree, &n2v_js[edge_offsets[dsti]],
                  &n2v_qs[edge_offsets[dsti]]);
    }
  }
  cout << endl << "Generating a corpus for negative samples.." << endl;
  neg_qs =
      static_cast<float *>(aligned_malloc(nv * sizeof(float), DEFAULT_ALIGN));
  neg_js = static_cast<int *>(aligned_malloc(nv * sizeof(int), DEFAULT_ALIGN));
  node_cnts =
      static_cast<float *>(aligned_malloc(nv * sizeof(float), DEFAULT_ALIGN));
  memset(neg_qs, 0, nv * sizeof(float));
#pragma omp parallel for num_threads(n_threads)
  for (int i = 0; i < nv * n_walks; i++) {
    int src = train_order[i % nv];
#pragma omp atomic
    neg_qs[src]++;
    if (degrees[src] == 0)
      continue;
    int lastedgeidx = irand(offsets[src], offsets[src + 1]);
    int lastnode = edges[lastedgeidx];
#pragma omp atomic
    neg_qs[lastnode]++;
    for (int j = 2; j < walk_length; j++) {
      if (degrees[lastnode] == 0)
        break;
      lastedgeidx =
          offsets[lastnode] + walker_draw(degrees[lastnode],
                                          &n2v_qs[edge_offsets[lastedgeidx]],
                                          &n2v_js[edge_offsets[lastedgeidx]]);
      lastnode = edges[lastedgeidx];
#pragma omp atomic
      neg_qs[lastnode]++;
    }
  }
  for (int i = 0; i < nv; i++)
    node_cnts[i] = neg_qs[i];
  float sum = 0;
  for (int i = 0; i < nv; i++) {
    neg_qs[i] = pow(neg_qs[i], 0.75f);
    sum += neg_qs[i];
  }
  for (int i = 0; i < nv; i++)
    neg_qs[i] *= nv / sum;
  init_walker(nv, neg_js, neg_qs);
  cout << endl;
  if (verbosity > 0)
#if VECTORIZE
    cout << "Using vectorized operations" << endl;
#else
    cout << "Not using vectorized operations (!)" << endl;
#endif



  // By Hao. Print RNCE related parameters
  // cout << "^^ RNCE buff ^^" << endl;
  // cout << "Regularized dist. func.: " << reg << endl;
  // cout << "Beta: " << beta << endl;
  // cout << "Upper bound of w: " << ub_w << endl;
  // cout << "Gamma(Laplacian/Gaussian kernel): " << gam << endl;
  cout << "dim: " << n_hidden << endl;
  cout << "^^ QIP included ^^" << endl;
  cout << "qip: " << qip << endl;
  cout << "num qubits: " << num_qubits << endl;
  cout << "sample times: " << sample_times << endl;
  cout << "mode: " << out_mode << endl;
  cout << endl;
  //



  auto begin = chrono::steady_clock::now();
  Train();
  auto end = chrono::steady_clock::now();
  if (verbosity > 0)
    cout << endl
         << "Calculations took "
         << chrono::duration_cast<chrono::duration<float>>(end - begin).count()
         << " s to run" << endl;
  /*
  ofstream output(embedding_file, ios::binary);
  output.write(reinterpret_cast<char *>(wVtx), sizeof(float) * n_hidden * nv);
  output.close();
  */

  // By Hao
  Output(embedding_file);
}
