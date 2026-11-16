# Revision-round statistics (plan: data/analysis_plan_round3.md)

## 4 boats
                  points     task  bonus
comm     policy                         
degraded A1      4096.94  2272.60   0.37
         A2      4787.93  2417.78   0.46
         L3      4667.72  2377.40   0.45
         P0      1399.42  1280.38   0.02
         P1      5182.87  2426.58   0.54
         P2      4844.46  2390.72   0.48
         P3      4396.62  2306.57   0.42
         P4      5223.05  2468.02   0.54
         P5      4146.70  2224.03   0.38
         R3      4153.63  2333.65   0.36
ideal    A1      4214.81  2309.20   0.38
         A2      4983.10  2399.07   0.51
         L3      4983.10  2399.07   0.51
         P0      2454.62  1570.83   0.18
         P1      5123.70  2395.78   0.54
         P2      4968.78  2381.77   0.51
         P3      4530.38  2375.46   0.42
         P4      4934.42  2398.53   0.49
         P5      4223.49  2215.09   0.40
         R3      4525.01  2357.47   0.42

yields per run (new policies):
comm      policy
degraded  L3        2.98
          P5        1.20
          R3        4.11
ideal     L3        1.37
          P5        0.20
          R3        3.08

## 2 boats
                  points     task  bonus
comm     policy                         
degraded A1      3577.92  2093.56   0.30
         A2      4073.02  2159.01   0.38
         L3      4061.40  2139.15   0.39
         P0       868.64   868.64   0.00
         P1      3863.81  2132.56   0.35
         P2      3848.42  2127.46   0.35
         P3      3752.53  2143.89   0.32
         P4      3916.21  2139.80   0.36
         P5      2686.78  1788.91   0.18
         R3      3864.64  2118.94   0.35
ideal    A1      3836.82  2160.64   0.34
         A2      4161.63  2187.95   0.40
         L3      4161.63  2187.95   0.40
         P0      1615.68  1161.01   0.10
         P1      3968.74  2150.64   0.36
         P2      3881.22  2119.50   0.36
         P3      4115.17  2191.30   0.39
         P4      4277.37  2184.40   0.42
         P5      2890.01  1891.39   0.20
         R3      4115.99  2168.97   0.40

yields per run (new policies):
comm      policy
degraded  L3        0.82
          P5        0.40
          R3        1.10
ideal     L3        0.32
          P5        0.07
          R3        0.76

### n4_lease_task_points
| contrast | outcome | mean [95% CI, unadjusted] | d_z | p | p_Holm | wins |
|---|---|---|---|---|---|---|
| L3-P3 | task_points | +71 [+20, +121] | +0.20 | 0.0063 | 0.0127 | 61% |
| L3-A2 | task_points | -40 [-81, -1] | -0.14 | 0.0471 | 0.0471 | 30% |

### n4_lease_ideal_task_points
| contrast | outcome | mean [95% CI, unadjusted] | d_z | p | p_Holm | wins |
|---|---|---|---|---|---|---|
| L3-P3 | task_points | +24 [-22, +69] | +0.07 | 0.3148 | 0.6296 | 56% |
| L3-A2 | task_points | +0 [+nan, +nan] | +nan | 1.0000 | 1.0000 | 0% |

### n4_lease_points
| contrast | outcome | mean [95% CI, unadjusted] | d_z | p | p_Holm | wins |
|---|---|---|---|---|---|---|
| L3-P3 | points | +271 [-125, +675] | +0.09 | 0.1904 | 0.3808 | 57% |
| L3-A2 | points | -120 [-485, +252] | -0.05 | 0.5249 | 0.5249 | 30% |

### n4_lease_ideal_points
| contrast | outcome | mean [95% CI, unadjusted] | d_z | p | p_Holm | wins |
|---|---|---|---|---|---|---|
| L3-P3 | points | +453 [+28, +893] | +0.14 | 0.0416 | 0.0833 | 55% |
| L3-A2 | points | +0 [+nan, +nan] | +nan | 1.0000 | 1.0000 | 0% |

### n4_task_points
| contrast | outcome | mean [95% CI, unadjusted] | d_z | p | p_Holm | wins |
|---|---|---|---|---|---|---|
| R3-P3 | task_points | +27 [-30, +81] | +0.07 | 0.3457 | 0.3457 | 58% |
| R3-A2 | task_points | -84 [-132, -36] | -0.24 | 0.0007 | 0.0022 | 32% |
| P4-P5 | task_points | +244 [+178, +312] | +0.50 | 0.0000 | 0.0000 | 71% |
| P5-P3 | task_points | -83 [-153, -14] | -0.16 | 0.0211 | 0.0422 | 48% |

### n4_points
| contrast | outcome | mean [95% CI, unadjusted] | d_z | p | p_Holm | wins |
|---|---|---|---|---|---|---|
| R3-P3 | points | -243 [-705, +230] | -0.07 | 0.3170 | 0.5684 | 54% |
| R3-A2 | points | -634 [-1068, -198] | -0.20 | 0.0047 | 0.0140 | 32% |
| P4-P5 | points | +1076 [+603, +1543] | +0.31 | 0.0000 | 0.0002 | 70% |
| P5-P3 | points | -250 [-700, +205] | -0.08 | 0.2842 | 0.5684 | 48% |

### n4_desc
        comm policy   points     task  bonus
0   degraded     A1  4096.94  2272.60   0.37
1   degraded     A2  4787.93  2417.78   0.46
2   degraded     L3  4667.72  2377.40   0.45
3   degraded     P0  1399.42  1280.38   0.02
4   degraded     P1  5182.87  2426.58   0.54
5   degraded     P2  4844.46  2390.72   0.48
6   degraded     P3  4396.62  2306.57   0.42
7   degraded     P4  5223.05  2468.02   0.54
8   degraded     P5  4146.70  2224.03   0.38
9   degraded     R3  4153.63  2333.65   0.36
10     ideal     A1  4214.81  2309.20   0.38
11     ideal     A2  4983.10  2399.07   0.51
12     ideal     L3  4983.10  2399.07   0.51
13     ideal     P0  2454.62  1570.83   0.18
14     ideal     P1  5123.70  2395.78   0.54
15     ideal     P2  4968.78  2381.77   0.51
16     ideal     P3  4530.38  2375.46   0.42
17     ideal     P4  4934.42  2398.53   0.49
18     ideal     P5  4223.49  2215.09   0.40
19     ideal     R3  4525.01  2357.47   0.42

### n2_lease_task_points
| contrast | outcome | mean [95% CI, unadjusted] | d_z | p | p_Holm | wins |
|---|---|---|---|---|---|---|
| L3-P3 | task_points | -5 [-65, +56] | -0.01 | 0.8787 | 0.8787 | 49% |
| L3-A2 | task_points | -20 [-58, +16] | -0.07 | 0.2943 | 0.5885 | 16% |

### n2_lease_ideal_task_points
| contrast | outcome | mean [95% CI, unadjusted] | d_z | p | p_Holm | wins |
|---|---|---|---|---|---|---|
| L3-P3 | task_points | -3 [-59, +49] | -0.01 | 0.9051 | 1.0000 | 39% |
| L3-A2 | task_points | +0 [+nan, +nan] | +nan | 1.0000 | 1.0000 | 0% |

### n2_lease_points
| contrast | outcome | mean [95% CI, unadjusted] | d_z | p | p_Holm | wins |
|---|---|---|---|---|---|---|
| L3-P3 | points | +309 [-87, +699] | +0.11 | 0.1286 | 0.2571 | 49% |
| L3-A2 | points | -12 [-269, +238] | -0.01 | 0.9318 | 0.9318 | 17% |

### n2_lease_ideal_points
| contrast | outcome | mean [95% CI, unadjusted] | d_z | p | p_Holm | wins |
|---|---|---|---|---|---|---|
| L3-P3 | points | +46 [-270, +367] | +0.02 | 0.7764 | 1.0000 | 40% |
| L3-A2 | points | +0 [+nan, +nan] | +nan | 1.0000 | 1.0000 | 0% |

### n2_task_points
| contrast | outcome | mean [95% CI, unadjusted] | d_z | p | p_Holm | wins |
|---|---|---|---|---|---|---|
| R3-P3 | task_points | -25 [-74, +23] | -0.07 | 0.3167 | 0.3167 | 44% |
| R3-A2 | task_points | -40 [-89, +10] | -0.11 | 0.1155 | 0.2311 | 28% |
| P4-P5 | task_points | +351 [+273, +431] | +0.61 | 0.0000 | 0.0000 | 78% |
| P5-P3 | task_points | -355 [-434, -279] | -0.63 | 0.0000 | 0.0000 | 30% |

### n2_points
| contrast | outcome | mean [95% CI, unadjusted] | d_z | p | p_Holm | wins |
|---|---|---|---|---|---|---|
| R3-P3 | points | +112 [-241, +477] | +0.04 | 0.5457 | 0.5457 | 44% |
| R3-A2 | points | -208 [-543, +127] | -0.09 | 0.2208 | 0.4416 | 30% |
| P4-P5 | points | +1229 [+812, +1648] | +0.41 | 0.0000 | 0.0000 | 79% |
| P5-P3 | points | -1066 [-1489, -658] | -0.35 | 0.0000 | 0.0000 | 31% |

### n2_desc
        comm policy   points     task  bonus
0   degraded     A1  3577.92  2093.56   0.30
1   degraded     A2  4073.02  2159.01   0.38
2   degraded     L3  4061.40  2139.15   0.39
3   degraded     P0   868.64   868.64   0.00
4   degraded     P1  3863.81  2132.56   0.35
5   degraded     P2  3848.42  2127.46   0.35
6   degraded     P3  3752.53  2143.89   0.32
7   degraded     P4  3916.21  2139.80   0.36
8   degraded     P5  2686.78  1788.91   0.18
9   degraded     R3  3864.64  2118.94   0.35
10     ideal     A1  3836.82  2160.64   0.34
11     ideal     A2  4161.63  2187.95   0.40
12     ideal     L3  4161.63  2187.95   0.40
13     ideal     P0  1615.68  1161.01   0.10
14     ideal     P1  3968.74  2150.64   0.36
15     ideal     P2  3881.22  2119.50   0.36
16     ideal     P3  4115.17  2191.30   0.39
17     ideal     P4  4277.37  2184.40   0.42
18     ideal     P5  2890.01  1891.39   0.20
19     ideal     R3  4115.99  2168.97   0.40

### robust_harsh_radio
| contrast | outcome | mean [95% CI, unadjusted] | d_z | p | p_Holm | wins |
|---|---|---|---|---|---|---|
| P4-P3 | task_points | +147 [+61, +243] | +0.31 | 0.0022 | nan | 63% |
| A2-P3 | task_points | +133 [+52, +216] | +0.32 | 0.0020 | nan | 71% |
| R3-P3 | task_points | +36 [-56, +120] | +0.08 | 0.4281 | nan | 61% |

### robust_harsh_radio_desc
  policy     task  yields  bonus
0     A2  2422.47    2.37   0.52
1     P3  2289.14    8.33   0.40
2     P4  2436.29    0.00   0.51
3     R3  2325.26    4.43   0.48

### robust_slow_boats
| contrast | outcome | mean [95% CI, unadjusted] | d_z | p | p_Holm | wins |
|---|---|---|---|---|---|---|
| P4-P3 | task_points | +218 [+130, +303] | +0.49 | 0.0000 | nan | 71% |
| A2-P3 | task_points | +157 [+68, +243] | +0.35 | 0.0006 | nan | 69% |
| R3-P3 | task_points | +35 [-57, +124] | +0.07 | 0.4578 | nan | 58% |

### robust_slow_boats_desc
  policy     task  yields  bonus
0     A2  2316.68    1.66   0.31
1     P3  2159.77    6.45   0.21
2     P4  2377.41    0.00   0.41
3     R3  2194.40    3.93   0.18

### n1_desc
  policy    points      task  bonus   home
0     A2  1756.148  1460.180   0.06  0.810
1     P2  1639.870  1405.188   0.05  0.805
