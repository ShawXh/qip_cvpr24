#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <torch/extension.h>

#include <iostream>
#include <iomanip>
#include <random>
#include <complex>
#include <cmath>

// #include <pybind11/pybind11.h>

#define float double

namespace py = pybind11;

using namespace std;

// std::mt19937 rng(time(0));
std::mt19937 rng(0);
std::uniform_real_distribution<double> randf(0., 1.);

void setRandSeed(int seed) {
  rng.seed((unsigned) seed);
}

double myrand() {
  double r = randf(rng);
  return r;
}

torch::Tensor thrand() {
  return torch::rand({1})[0];
}

int fast_pow2(int n) {
  return 1 << n;
}

float ComputeProbHand(int idx, int num_states, float phi) {
  complex<double> tmp;
  tmp = 0. + 0i;
  for (int k = 0; k < num_states; k++)
      tmp += exp(2i * M_PI * (double)k * (-idx / (double)num_states + phi));
  return (tmp * conj(tmp)).real() / ((float)num_states * (float)num_states);
}

float ComputeProbTh(
    int idx, 
    int num_states, 
    float phi) {
  auto opt = torch::TensorOptions()
    .dtype(torch::kFloat32)
    .requires_grad(false);
  auto val = torch::arange(0, num_states, opt);
  val = 2 * M_PI * (-idx / (double)num_states + phi) * val;
  float re = torch::cos(val).sum().item().toFloat();
  float im = torch::sin(val).sum().item().toFloat();
  return (re * re + im * im) / ((float)num_states * (float)num_states);
}

// float ComputeProbTh(
//     torch::Tensor val_cache, 
//     int idx, 
//     int num_states, 
//     float phi) {
//   // val is from the cache, which is set as torch::arange(0, num_states, opt)
//   // before calling the function
//   val_cache = 2 * M_PI * (-idx / (double)num_states + phi) * val_cache;
//   float re = torch::cos(val_cache).sum().item().toFloat();
//   float im = torch::sin(val_cache).sum().item().toFloat();
//   return (re * re + im * im) / ((float)num_states * (float)num_states);
// }

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

// float FastSampleQPEOnce(int num_qubits, const float gt_ip, torch::Tensor val_cache, bool debug=false) {
//   if (debug) {
//     cout << num_qubits << " Quibits" << endl;
//     cout << "Ground truth: " << gt_ip << endl;
//   }
//   if (gt_ip > 1.) return 1.;
//   if (gt_ip < -1.) return -1.;

//   float phi = 0.5 * acos(-gt_ip) / M_PI;
//   if (debug) cout << "gt R: " << phi << endl;

//   int num_states = fast_pow2(num_qubits);
//   if (debug) cout << num_states << " states" << endl;

//   // fast sampling
//   float p = randf(rng);
//   if (debug) cout << "random number p: " << p << endl;

//   float p1 = 0., p2 = 0.;
//   int idx1 = 0, idx2 = 0;

//   idx1 = (int)ceil(num_states * phi) % num_states;
//   p1 = ComputeProbTh(val_cache, idx1, num_states, phi);
//   if (debug) cout << "idx1: " << idx1 << ", p1: " << p1 << endl;
//   p -= p1;
//   if (p < 0) return -cos(2 * M_PI * idx1 / num_states);
//   idx2 = (int)floor(num_states * phi);
//   p2 = ComputeProbTh(val_cache, idx2, num_states, phi);
//   if (debug) cout << "idx2: " << idx2 << ", p2: " << p2 << endl;
//   p -= p2;
//   if (p < 0) return -cos(2 * M_PI * idx2 / num_states);

//   // other cases
//   for (int i = 0; i < num_states / 2 - 1; i++) {
//     // idx1
//     idx1 = (idx1 + 1) % num_states;
//     p1 = ComputeProbTh(val_cache, idx1, num_states, phi);
//     if (debug) cout << "idx1: " << idx1 << ", p1: " << p1 << endl;
//     p -= p1;
//     if (p < 0) return -cos(2 * M_PI * (float)idx1 / num_states);
//     // idx2
//     idx2 = ((idx2 - 1) + num_states) % num_states;
//     p2 = ComputeProbTh(val_cache, idx2, num_states, phi);
//     if (debug) cout << "idx2: " << idx2 << ", p2: " << p2 << endl;
//     p -= p2;
//     if (p < 0) return -cos(2 * M_PI * (float)idx2 / num_states);
//   }

//   cout << "ERROR: Failed in QPE " << endl;
//   exit(1);
//   return 0;
// }

void RecordIdx(unordered_map<int, int> &idx_count, const int idx) {
  if (idx_count.find(idx) == idx_count.end()) idx_count[idx] = 1;
  else (idx_count[idx])++;
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

// float GetProb(
//     torch::Tensor &val_cache,
//     std::vector<float> &probs, 
//     int &idx, 
//     int &compute_iter, 
//     const int num_states, 
//     const float phi) {
//   float prob;
//   if (probs.size() < compute_iter + 1) {
//     prob = ComputeProbTh(val_cache, idx, num_states, phi);
//     probs.push_back(prob);
//   } else {
//     prob = probs[compute_iter];
//     // cout << "fetch probs[" << compute_iter << "]: " << prob << endl;
//   }
//   return prob;
// }

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

// float FastSampleQPECache(int num_qubits, float gt_ip, torch::Tensor val_cache, int sample_times=0, bool debug=false, const string out_mode) {
  
//   if (sample_times == 0) return FastSampleQPEOnce(num_qubits, gt_ip, val_cache, debug);

//   sample_times = sample_times * 2 + 1;

//   if (debug) {
//     cout << num_qubits << " Quibits" << endl;
//     cout << "Ground truth: " << gt_ip << endl;
//   }
//   if (gt_ip > 1.) return 1.;
//   if (gt_ip < -1.) return -1.;

//   float gt_r = 0.5 * acos(-gt_ip) / M_PI;
//   if (debug) cout << "gt R: " << gt_r << endl;

//   int num_states = fast_pow2(num_qubits);
//   if (debug) cout << num_states << " states" << endl;

//   // fast sampling
//   float p;
  

//   float p1 = 0., p2 = 0.;
//   int idx1 = 0, idx2 = 0;
//   int compute_iter = 0;

//   std::unordered_map<int, int> idx_count;
//   idx_count.reserve(sample_times);
//   std::vector<float> probs;

//   for (int iter = 0; iter < sample_times; iter++) {
//     p = randf(rng);
//     if (debug) cout << "Sample iteration " << iter << ", random number p: " << p << endl;
//     compute_iter = 0;

//     idx1 = (int)ceil(num_states * gt_r) % num_states;
//     p1 = GetProb(val_cache, probs, idx1, compute_iter, num_states, gt_r);
//     if (debug) cout << "idx1: " << idx1 << ", p1: " << p1 << endl;
//     compute_iter++;
//     p -= p1;
//     if (p < 0) {
//       RecordIdx(idx_count, idx1);
//       continue;
//     }

//     idx2 = (int)floor(num_states * gt_r);
//     p2 = GetProb(val_cache, probs, idx2, compute_iter, num_states, gt_r);
//     if (debug) cout << "idx2: " << idx2 << ", p2: " << p2 << endl;
//     compute_iter++;
//     p -= p2;
//     if (p < 0) {
//       RecordIdx(idx_count, idx2);
//       continue;
//     }

//     // other cases
//     for (int i = 0; i < num_states / 2 - 1; i++) {
//       // idx1
//       idx1 = (idx1 + 1) % num_states;
//       p1 = GetProb(val_cache, probs, idx1, compute_iter, num_states, gt_r);
//       if (debug) cout << "idx1: " << idx1 << ", p1: " << p1 << endl;
//       compute_iter++;
//       p -= p1;
//       if (p < 0) {
//         RecordIdx(idx_count, idx1);
//         break;
//       }

//       // idx2
//       idx2 = ((idx2 - 1) + num_states) % num_states;
//       p2 = GetProb(val_cache, probs, idx2, compute_iter, num_states, gt_r);
//       if (debug) cout << "idx2: " << idx2 << ", p2: " << p2 << endl;
//       compute_iter++;
//       p -= p2;
//       if (p < 0) {
//         RecordIdx(idx_count, idx2);
//         break;
//       }
//     }
//   }

//   int idx, max_time = 0;
//   for (auto i = idx_count.cbegin(); i != idx_count.cend(); i++) {
//     if (i->second > max_time) {max_time = i->second; idx = i->first;}
//   }

//   if (debug) cout << "Final idx: " << idx << endl;
//   return -cos(2 * M_PI * (float)idx / num_states);
// }

float SampleQPE(int num_qubits, float gt_ip, int sample_times=0, bool debug=false, char *out_mode="mode") {
  sample_times = sample_times * 2 + 1;

  if (debug) {
    cout << "Output mode: " << out_mode << endl;
    cout << num_qubits << " Quibits" << endl;
    cout << "Ground truth: " << gt_ip << endl;
  }
  if (gt_ip > 1.) return 1.;
  if (gt_ip < -1.) return -1.;

  float phi = 0.5 * acos(-gt_ip) / M_PI;
  if (debug) cout << "phi: " << phi << endl;

  int num_states = fast_pow2(num_qubits);
  if (debug) cout << num_states << " states" << endl;

  // all cases
  float *all_probs = new float [num_states];
  for (int s = 0; s < num_states; s++) all_probs[s] = ComputeProbHand(s, num_states, phi);
  // for (int s = 0; s < num_states; s++) all_probs[s] = ComputeProbTh(s, num_states, phi);
  
  // sampling
  float p;
  float p1 = 0., p2 = 0.;
  int idx1 = 0, idx2 = 0;

  std::unordered_map<int, int> idx_count;
  idx_count.reserve(sample_times * 10);

  for (int iter = 0; iter < sample_times; iter++) {
    p = randf(rng);
    if (debug) cout << "Sample iteration " << iter << ", random number p: " << p << endl;

    idx1 = (int)ceil(num_states * phi) % num_states;
    p1 = all_probs[idx1]; 
    if (debug) cout << "idx1: " << idx1 << ", p1: " << p1 << endl;
    p -= p1;
    if (p < 0) {
      RecordIdx(idx_count, idx1);
      continue;
    }

    idx2 = (int)floor(num_states * phi);
    p2 = all_probs[idx2];
    if (debug) cout << "idx2: " << idx2 << ", p2: " << p2 << endl;
    p -= p2;
    if (p < 0) {
      RecordIdx(idx_count, idx2);
      continue;
    }

    // other cases
    for (int i = 0; i < num_states / 2 - 1; i++) {
      // idx1
      idx1 = (idx1 + 1) % num_states;
      p1 = all_probs[idx1]; 
      if (debug) cout << "idx1: " << idx1 << ", p1: " << p1 << endl;
      p -= p1;
      if (p < 0) {
        RecordIdx(idx_count, idx1);
        break;
      }

      // idx2
      idx2 = ((idx2 - 1) + num_states) % num_states;
      p2 = all_probs[idx2];
      if (debug) cout << "idx2: " << idx2 << ", p2: " << p2 << endl;
      p -= p2;
      if (p < 0) {
        RecordIdx(idx_count, idx2);
        break;
      }
    }

  }

  // output
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

torch::Tensor FastSampleQPETensor(torch::Tensor x, int num_qubits, const int sample_times, char *out_mode="mode") {
  auto size = x.sizes();
  x = x.view({-1});
  int length = (int)x.size(0);
  #pragma omp parallel for
  for (int i = 0; i < length; i++) 
    x[i] = FastSampleQPE(num_qubits, x[i].item().toFloat(), sample_times, false, out_mode);
  x = x.view(size);
  return x;
}

torch::Tensor FastSampleQPETensorAllCase(torch::Tensor x, int num_qubits, const int sample_times, char *out_mode="mode") {
  auto size = x.sizes();
  x = x.view({-1});
  int length = (int)x.size(0);
  #pragma omp parallel for
  for (int i = 0; i < length; i++) 
    x[i] = SampleQPE(num_qubits, x[i].item().toFloat(), sample_times, false, out_mode);
  x = x.view(size);
  return x;
}

// torch::Tensor FastSampleQPETensorCache(torch::Tensor x, torch::Tensor val_cache, int num_qubits, const int sample_times) {
//   auto size = x.sizes();
//   x = x.view({-1});
//   int length = (int)x.size(0);
//   for (int i = 0; i < length; i++) 
//     x[i] = FastSampleQPECache(num_qubits, x[i].item().toFloat(), val_cache, sample_times, false);
//   x = x.view(size);
//   return x;
// }

// torch::Tensor SwapTestTensor(torch::Tensor x, int num_qubits, const int sample_times) {
//   auto size = x.sizes();
//   x = x.view({-1});
//   int length = (int)x.size(0);
//   #pragma omp parallel for
//   for (int i = 0; i < length; i++) 
//     x[i] = FastSampleQPE(num_qubits, x[i].item().toFloat(), sample_times, false);
//   x = x.view(size);
//   return x;
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myrand", &myrand, "rand");
  m.def("thrand", &thrand, "torch rand");
  m.def("elementqip", &FastSampleQPE, "sample from a quantum phase");
  // m.def("elementqip_cache", &FastSampleQPECache, "sample from a quantum phase");
  m.def("qip", &FastSampleQPETensor, "transforming tensors by quantum phase estimation");
  m.def("qip_all_case", &FastSampleQPETensorAllCase, "transforming tensors by quantum phase estimation");
  // m.def("qip_cache", &FastSampleQPETensorCache, "transforming tensors by quantum phase estimation");
  m.def("set_seed", &setRandSeed, "set random seed");
}

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}