//===----------------------------------------------------------------------===//
//
// Copyright the CYCLE LAB.
// All rights reserved.
//
//===----------------------------------------------------------------------===//
void test_lif(float v1[2][2], float v2[2][2], float v3[2][2]) {
  size_t v4 = 1;
  size_t v5 = 2;
  size_t v6 = 0;
  float v7 = 0.0e+00f;
  float v8 = 1.000000000e+00f;
  float v9 = 9.900000090e-01f;
  float v10[1][1];
  float v11[1][1];
  float v12[1][1];
  float v13[1][1];
  for (size_t v14 = v6; v14 < v5; v14 += v4) {
    for (size_t v15 = v6; v15 < v5; v15 += v4) {
      v12[v6][v6] = v9;
      float v16;
      v16 = v1[v14][v15];
      float v17;
      v17 = v12[v6][v6];
      float v18 = v16 * v17;
      v13[v6][v6] = v18;
      float v19;
      v19 = v2[v14][v15];
      float v20;
      v20 = v13[v6][v6];
      float v21 = v19 + v20;
      v10[v6][v6] = v21;
      v11[v6][v6] = v8;
      float v22;
      v22 = v11[v6][v6];
      float v23;
      v23 = v10[v6][v6];
      bool v24 = v23 < v22;
      bool v25 = v23 == v23;
      bool v26 = v22 == v22;
      bool v27 = v25 && v26;
      bool v28 = v27 && v24;
      float v29 = v28 ? v7 : v8;
      v3[v14][v15] = v29;
    };
  }
  return;
}
