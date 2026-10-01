#define RANDOM_FOREST_NODE_COUNT 3456
#define RANDOM_FOREST_MODEL_BYTES 27648  /* estimated: node count * 8 bytes/node */



    // !!! This file is generated using emlearn !!!

    #include <stdint.h>
    

static inline int32_t random_forest_tree_0(const float *features, int32_t features_length) {
          if (features[12] < -0.103503f) {
              if (features[9] < -0.367172f) {
                  if (features[2] < -0.891358f) {
                      if (features[13] < -0.102865f) {
                          if (features[3] < 1.282808f) {
                              if (features[15] < -0.088272f) {
                                  if (features[5] < -0.126582f) {
                                      return 2;
                                  } else {
                                      if (features[11] < 0.511574f) {
                                          if (features[7] < 0.480436f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      } else {
                                          if (features[3] < 0.759286f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      }
                                  }
                              } else {
                                  return 2;
                              }
                          } else {
                              if (features[8] < 0.308626f) {
                                  if (features[8] < 0.235533f) {
                                      if (features[10] < 0.689215f) {
                                          return 0;
                                      } else {
                                          return 2;
                                      }
                                  } else {
                                      return 2;
                                  }
                              } else {
                                  if (features[6] < 0.267265f) {
                                      return 1;
                                  } else {
                                      return 3;
                                  }
                              }
                          }
                      } else {
                          if (features[8] < 0.679628f) {
                              if (features[5] < 0.233651f) {
                                  if (features[10] < 0.266141f) {
                                      return 2;
                                  } else {
                                      if (features[11] < 0.402778f) {
                                          if (features[10] < 0.698419f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      } else {
                                          if (features[10] < 0.394858f) {
                                              if (features[8] < 0.283885f) {
                                                  if (features[15] < -0.437214f) {
                                                      return 2;
                                                  } else {
                                                      return 2;
                                                  }
                                              } else {
                                                  if (features[12] < -0.250225f) {
                                                      if (features[1] < 0.517233f) {
                                                          return 2;
                                                      } else {
                                                          return 2;
                                                      }
                                                  } else {
                                                      return 2;
                                                  }
                                              }
                                          } else {
                                              if (features[13] < 0.005743f) {
                                                  return 2;
                                              } else {
                                                  if (features[1] < 0.499177f) {
                                                      if (features[14] < 0.787238f) {
                                                          return 2;
                                                      } else {
                                                          return 2;
                                                      }
                                                  } else {
                                                      if (features[4] < 2.029360f) {
                                                          return 2;
                                                      } else {
                                                          return 2;
                                                      }
                                                  }
                                              }
                                          }
                                      }
                                  }
                              } else {
                                  if (features[7] < 0.491518f) {
                                      return 2;
                                  } else {
                                      return 2;
                                  }
                              }
                          } else {
                              return 2;
                          }
                      }
                  } else {
                      if (features[12] < -0.162882f) {
                          if (features[14] < 0.608841f) {
                              if (features[4] < 1.392960f) {
                                  if (features[8] < 0.135072f) {
                                      return 2;
                                  } else {
                                      return 0;
                                  }
                              } else {
                                  if (features[1] < 0.391569f) {
                                      if (features[5] < 0.031467f) {
                                          if (features[10] < 0.351684f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          return 1;
                                      }
                                  } else {
                                      if (features[8] < 0.253418f) {
                                          if (features[7] < 0.348534f) {
                                              if (features[10] < 0.349024f) {
                                                  return 0;
                                              } else {
                                                  return 2;
                                              }
                                          } else {
                                              return 2;
                                          }
                                      } else {
                                          return 1;
                                      }
                                  }
                              }
                          } else {
                              if (features[8] < 0.182230f) {
                                  if (features[8] < 0.148305f) {
                                      if (features[1] < 0.350637f) {
                                          return 2;
                                      } else {
                                          return 2;
                                      }
                                  } else {
                                      return 2;
                                  }
                              } else {
                                  if (features[5] < -0.051602f) {
                                      return 2;
                                  } else {
                                      if (features[2] < -0.711465f) {
                                          return 2;
                                      } else {
                                          return 1;
                                      }
                                  }
                              }
                          }
                      } else {
                          if (features[4] < 2.197675f) {
                              if (features[10] < 0.327049f) {
                                  return 0;
                              } else {
                                  return 2;
                              }
                          } else {
                              if (features[4] < 2.344076f) {
                                  return 3;
                              } else {
                                  if (features[10] < 0.439694f) {
                                      return 3;
                                  } else {
                                      return 3;
                                  }
                              }
                          }
                      }
                  }
              } else {
                  if (features[10] < 0.285220f) {
                      if (features[8] < 0.276292f) {
                          if (features[10] < 0.101947f) {
                              if (features[13] < 0.268874f) {
                                  if (features[2] < -0.196061f) {
                                      return 0;
                                  } else {
                                      return 2;
                                  }
                              } else {
                                  return 1;
                              }
                          } else {
                              if (features[0] < 0.026955f) {
                                  if (features[15] < -0.270889f) {
                                      if (features[15] < -0.670729f) {
                                          if (features[10] < 0.153476f) {
                                              return 1;
                                          } else {
                                              if (features[1] < 0.188901f) {
                                                  return 1;
                                              } else {
                                                  return 1;
                                              }
                                          }
                                      } else {
                                          if (features[2] < -0.436890f) {
                                              return 2;
                                          } else {
                                              if (features[4] < 0.947581f) {
                                                  return 1;
                                              } else {
                                                  if (features[1] < 0.212331f) {
                                                      return 0;
                                                  } else {
                                                      return 1;
                                                  }
                                              }
                                          }
                                      }
                                  } else {
                                      if (features[1] < 0.171469f) {
                                          return 0;
                                      } else {
                                          if (features[0] < -0.007281f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      }
                                  }
                              } else {
                                  if (features[5] < 0.017034f) {
                                      return 2;
                                  } else {
                                      return 2;
                                  }
                              }
                          }
                      } else {
                          if (features[14] < 0.234196f) {
                              if (features[15] < -0.149428f) {
                                  if (features[8] < 0.395034f) {
                                      return 1;
                                  } else {
                                      if (features[1] < 0.296622f) {
                                          if (features[12] < -0.173975f) {
                                              return 1;
                                          } else {
                                              return 1;
                                          }
                                      } else {
                                          if (features[15] < -0.320348f) {
                                              return 1;
                                          } else {
                                              return 0;
                                          }
                                      }
                                  }
                              } else {
                                  if (features[5] < -0.059232f) {
                                      return 3;
                                  } else {
                                      if (features[8] < 0.394035f) {
                                          return 0;
                                      } else {
                                          return 0;
                                      }
                                  }
                              }
                          } else {
                              if (features[12] < -0.187895f) {
                                  if (features[2] < -0.718145f) {
                                      if (features[9] < -0.257446f) {
                                          return 1;
                                      } else {
                                          return 1;
                                      }
                                  } else {
                                      return 1;
                                  }
                              } else {
                                  return 1;
                              }
                          }
                      }
                  } else {
                      if (features[8] < 0.457632f) {
                          if (features[3] < 1.021767f) {
                              if (features[6] < 0.036003f) {
                                  return 1;
                              } else {
                                  if (features[13] < 0.017081f) {
                                      if (features[15] < -0.576075f) {
                                          return 1;
                                      } else {
                                          if (features[2] < -0.525750f) {
                                              if (features[0] < 0.029274f) {
                                                  return 2;
                                              } else {
                                                  if (features[15] < -0.376157f) {
                                                      if (features[11] < 0.629630f) {
                                                          return 2;
                                                      } else {
                                                          return 2;
                                                      }
                                                  } else {
                                                      return 2;
                                                  }
                                              }
                                          } else {
                                              return 0;
                                          }
                                      }
                                  } else {
                                      if (features[9] < -0.292050f) {
                                          if (features[6] < 0.079718f) {
                                              return 2;
                                          } else {
                                              if (features[4] < 1.843806f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          }
                                      } else {
                                          if (features[15] < -0.545640f) {
                                              return 2;
                                          } else {
                                              if (features[4] < 1.085930f) {
                                                  if (features[8] < 0.139817f) {
                                                      return 2;
                                                  } else {
                                                      return 2;
                                                  }
                                              } else {
                                                  if (features[12] < -0.325930f) {
                                                      if (features[3] < 0.630967f) {
                                                          return 2;
                                                      } else {
                                                          return 2;
                                                      }
                                                  } else {
                                                      return 2;
                                                  }
                                              }
                                          }
                                      }
                                  }
                              }
                          } else {
                              if (features[4] < 1.872560f) {
                                  if (features[1] < 0.278081f) {
                                      return 0;
                                  } else {
                                      return 0;
                                  }
                              } else {
                                  if (features[14] < 0.057859f) {
                                      return 0;
                                  } else {
                                      return 1;
                                  }
                              }
                          }
                      } else {
                          if (features[12] < -0.167695f) {
                              return 1;
                          } else {
                              return 0;
                          }
                      }
                  }
              }
          } else {
              if (features[1] < 0.459991f) {
                  if (features[5] < -0.070436f) {
                      if (features[15] < -0.303208f) {
                          if (features[9] < -0.330594f) {
                              return 1;
                          } else {
                              return 1;
                          }
                      } else {
                          if (features[8] < 0.366905f) {
                              if (features[5] < -0.105572f) {
                                  return 3;
                              } else {
                                  if (features[0] < -0.055073f) {
                                      return 0;
                                  } else {
                                      if (features[15] < 0.127150f) {
                                          return 0;
                                      } else {
                                          return 0;
                                      }
                                  }
                              }
                          } else {
                              if (features[0] < -0.011459f) {
                                  if (features[6] < 0.123145f) {
                                      return 0;
                                  } else {
                                      if (features[4] < 2.607761f) {
                                          return 3;
                                      } else {
                                          return 3;
                                      }
                                  }
                              } else {
                                  return 0;
                              }
                          }
                      }
                  } else {
                      if (features[15] < -0.264734f) {
                          if (features[0] < -0.048844f) {
                              if (features[2] < -0.709589f) {
                                  return 2;
                              } else {
                                  return 2;
                              }
                          } else {
                              if (features[2] < -0.313397f) {
                                  if (features[8] < 0.114301f) {
                                      return 0;
                                  } else {
                                      if (features[11] < 0.798611f) {
                                          if (features[15] < -0.320706f) {
                                              if (features[5] < 0.003480f) {
                                                  if (features[12] < -0.016020f) {
                                                      return 1;
                                                  } else {
                                                      return 1;
                                                  }
                                              } else {
                                                  if (features[15] < -0.433059f) {
                                                      return 1;
                                                  } else {
                                                      return 1;
                                                  }
                                              }
                                          } else {
                                              return 1;
                                          }
                                      } else {
                                          return 1;
                                      }
                                  }
                              } else {
                                  return 0;
                              }
                          }
                      } else {
                          if (features[2] < -0.492413f) {
                              if (features[5] < 0.088302f) {
                                  if (features[0] < -0.072516f) {
                                      if (features[10] < 0.152226f) {
                                          return 0;
                                      } else {
                                          return 2;
                                      }
                                  } else {
                                      if (features[14] < 0.141232f) {
                                          if (features[10] < 0.467958f) {
                                              if (features[8] < 0.093279f) {
                                                  if (features[1] < 0.252260f) {
                                                      return 0;
                                                  } else {
                                                      return 2;
                                                  }
                                              } else {
                                                  if (features[4] < 1.054510f) {
                                                      if (features[1] < 0.182747f) {
                                                          return 0;
                                                      } else {
                                                          return 2;
                                                      }
                                                  } else {
                                                      if (features[6] < 0.188342f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              }
                                          } else {
                                              if (features[14] < 0.027199f) {
                                                  return 0;
                                              } else {
                                                  return 3;
                                              }
                                          }
                                      } else {
                                          if (features[0] < 0.043768f) {
                                              if (features[8] < 0.430508f) {
                                                  if (features[15] < -0.093524f) {
                                                      return 0;
                                                  } else {
                                                      return 0;
                                                  }
                                              } else {
                                                  if (features[5] < -0.037464f) {
                                                      return 0;
                                                  } else {
                                                      return 1;
                                                  }
                                              }
                                          } else {
                                              return 2;
                                          }
                                      }
                                  }
                              } else {
                                  return 3;
                              }
                          } else {
                              if (features[7] < 0.181016f) {
                                  if (features[11] < 0.886574f) {
                                      if (features[4] < 2.137038f) {
                                          if (features[0] < 0.042772f) {
                                              if (features[5] < -0.043924f) {
                                                  if (features[9] < -0.227947f) {
                                                      if (features[5] < -0.045244f) {
                                                          return 0;
                                                      } else {
                                                          return 3;
                                                      }
                                                  } else {
                                                      if (features[12] < -0.048274f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              } else {
                                                  if (features[13] < 0.136096f) {
                                                      return 0;
                                                  } else {
                                                      if (features[4] < 1.490906f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              }
                                          } else {
                                              if (features[7] < 0.172406f) {
                                                  return 0;
                                              } else {
                                                  return 1;
                                              }
                                          }
                                      } else {
                                          return 3;
                                      }
                                  } else {
                                      return 3;
                                  }
                              } else {
                                  if (features[4] < 2.078130f) {
                                      if (features[14] < 0.104632f) {
                                          return 0;
                                      } else {
                                          return 0;
                                      }
                                  } else {
                                      if (features[13] < -0.027208f) {
                                          if (features[13] < -0.140827f) {
                                              if (features[15] < -0.012806f) {
                                                  return 3;
                                              } else {
                                                  return 3;
                                              }
                                          } else {
                                              return 3;
                                          }
                                      } else {
                                          return 0;
                                      }
                                  }
                              }
                          }
                      }
                  }
              } else {
                  if (features[2] < -0.979514f) {
                      if (features[14] < 0.085280f) {
                          if (features[15] < -0.138289f) {
                              return 1;
                          } else {
                              if (features[5] < -0.000908f) {
                                  if (features[10] < 0.617662f) {
                                      return 0;
                                  } else {
                                      if (features[1] < 0.640207f) {
                                          return 2;
                                      } else {
                                          if (features[15] < -0.030480f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      }
                                  }
                              } else {
                                  if (features[10] < 0.455246f) {
                                      return 0;
                                  } else {
                                      return 3;
                                  }
                              }
                          }
                      } else {
                          if (features[0] < -0.102192f) {
                              return 2;
                          } else {
                              if (features[13] < 0.111804f) {
                                  return 2;
                              } else {
                                  return 2;
                              }
                          }
                      }
                  } else {
                      if (features[11] < 0.506944f) {
                          if (features[5] < -0.047160f) {
                              return 2;
                          } else {
                              if (features[13] < -0.289610f) {
                                  return 0;
                              } else {
                                  return 0;
                              }
                          }
                      } else {
                          if (features[5] < -0.038920f) {
                              if (features[8] < 0.283124f) {
                                  return 2;
                              } else {
                                  if (features[7] < 0.315376f) {
                                      if (features[6] < 0.221798f) {
                                          return 3;
                                      } else {
                                          return 3;
                                      }
                                  } else {
                                      if (features[3] < 1.396627f) {
                                          return 3;
                                      } else {
                                          if (features[1] < 0.512481f) {
                                              return 3;
                                          } else {
                                              if (features[15] < 0.005667f) {
                                                  return 3;
                                              } else {
                                                  return 3;
                                              }
                                          }
                                      }
                                  }
                              }
                          } else {
                              if (features[2] < -0.742415f) {
                                  if (features[3] < 1.931567f) {
                                      if (features[4] < 2.550276f) {
                                          return 3;
                                      } else {
                                          return 3;
                                      }
                                  } else {
                                      return 0;
                                  }
                              } else {
                                  if (features[15] < 0.047856f) {
                                      return 0;
                                  } else {
                                      return 3;
                                  }
                              }
                          }
                      }
                  }
              }
          }
        }
        

static inline int32_t random_forest_tree_1(const float *features, int32_t features_length) {
          if (features[10] < 0.413730f) {
              if (features[12] < -0.123089f) {
                  if (features[11] < 0.587963f) {
                      if (features[0] < -0.068823f) {
                          if (features[5] < -0.148440f) {
                              return 2;
                          } else {
                              if (features[15] < -0.625121f) {
                                  return 2;
                              } else {
                                  if (features[5] < -0.051560f) {
                                      if (features[1] < 0.523905f) {
                                          return 2;
                                      } else {
                                          return 2;
                                      }
                                  } else {
                                      if (features[5] < -0.020344f) {
                                          return 2;
                                      } else {
                                          return 2;
                                      }
                                  }
                              }
                          }
                      } else {
                          if (features[9] < -0.408594f) {
                              if (features[3] < 1.496695f) {
                                  if (features[8] < 0.146543f) {
                                      return 2;
                                  } else {
                                      if (features[8] < 0.255911f) {
                                          if (features[3] < 0.843337f) {
                                              return 2;
                                          } else {
                                              if (features[14] < 0.672431f) {
                                                  return 0;
                                              } else {
                                                  return 0;
                                              }
                                          }
                                      } else {
                                          return 2;
                                      }
                                  }
                              } else {
                                  if (features[15] < -0.201521f) {
                                      return 1;
                                  } else {
                                      return 3;
                                  }
                              }
                          } else {
                              if (features[10] < 0.086717f) {
                                  if (features[15] < -0.385284f) {
                                      return 1;
                                  } else {
                                      if (features[13] < 0.084221f) {
                                          return 0;
                                      } else {
                                          return 0;
                                      }
                                  }
                              } else {
                                  if (features[12] < -0.146833f) {
                                      if (features[10] < 0.334601f) {
                                          if (features[2] < -0.580890f) {
                                              if (features[3] < 0.944460f) {
                                                  if (features[7] < 0.166983f) {
                                                      return 1;
                                                  } else {
                                                      if (features[9] < -0.297265f) {
                                                          return 2;
                                                      } else {
                                                          return 1;
                                                      }
                                                  }
                                              } else {
                                                  if (features[0] < 0.026116f) {
                                                      if (features[12] < -0.197915f) {
                                                          return 1;
                                                      } else {
                                                          return 1;
                                                      }
                                                  } else {
                                                      return 1;
                                                  }
                                              }
                                          } else {
                                              if (features[6] < 0.088325f) {
                                                  if (features[12] < -0.418632f) {
                                                      if (features[14] < 0.644406f) {
                                                          return 1;
                                                      } else {
                                                          return 1;
                                                      }
                                                  } else {
                                                      if (features[15] < -0.389102f) {
                                                          return 1;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              } else {
                                                  return 3;
                                              }
                                          }
                                      } else {
                                          if (features[15] < -0.385509f) {
                                              return 2;
                                          } else {
                                              return 0;
                                          }
                                      }
                                  } else {
                                      if (features[13] < -0.158786f) {
                                          return 3;
                                      } else {
                                          if (features[3] < 1.694854f) {
                                              if (features[3] < 1.040664f) {
                                                  return 0;
                                              } else {
                                                  if (features[15] < -0.117662f) {
                                                      return 1;
                                                  } else {
                                                      return 1;
                                                  }
                                              }
                                          } else {
                                              return 0;
                                          }
                                      }
                                  }
                              }
                          }
                      }
                  } else {
                      if (features[1] < 0.174536f) {
                          return 2;
                      } else {
                          if (features[13] < -0.404212f) {
                              return 2;
                          } else {
                              if (features[9] < -0.165867f) {
                                  if (features[4] < 1.315511f) {
                                      return 2;
                                  } else {
                                      return 2;
                                  }
                              } else {
                                  return 2;
                              }
                          }
                      }
                  }
              } else {
                  if (features[11] < 0.525463f) {
                      if (features[8] < 0.368736f) {
                          if (features[4] < 0.525673f) {
                              return 3;
                          } else {
                              if (features[15] < -0.150454f) {
                                  if (features[10] < 0.279826f) {
                                      if (features[3] < 0.990669f) {
                                          if (features[14] < -0.205997f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          if (features[15] < -0.281756f) {
                                              return 1;
                                          } else {
                                              return 0;
                                          }
                                      }
                                  } else {
                                      if (features[6] < 0.144166f) {
                                          if (features[2] < -0.605977f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      } else {
                                          return 2;
                                      }
                                  }
                              } else {
                                  if (features[5] < -0.105338f) {
                                      return 3;
                                  } else {
                                      if (features[8] < 0.140509f) {
                                          if (features[8] < 0.139678f) {
                                              if (features[1] < 0.183248f) {
                                                  return 0;
                                              } else {
                                                  if (features[1] < 0.187172f) {
                                                      return 2;
                                                  } else {
                                                      if (features[0] < -0.038229f) {
                                                          return 2;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              }
                                          } else {
                                              return 3;
                                          }
                                      } else {
                                          if (features[12] < -0.091375f) {
                                              if (features[7] < 0.237676f) {
                                                  if (features[5] < -0.022531f) {
                                                      return 0;
                                                  } else {
                                                      if (features[12] < -0.108580f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              } else {
                                                  return 2;
                                              }
                                          } else {
                                              if (features[11] < 0.516204f) {
                                                  return 0;
                                              } else {
                                                  if (features[5] < 0.042647f) {
                                                      return 0;
                                                  } else {
                                                      return 0;
                                                  }
                                              }
                                          }
                                      }
                                  }
                              }
                          }
                      } else {
                          if (features[1] < 0.306517f) {
                              if (features[4] < 2.060637f) {
                                  if (features[2] < -0.851818f) {
                                      if (features[0] < 0.007820f) {
                                          return 1;
                                      } else {
                                          if (features[13] < 0.001557f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      }
                                  } else {
                                      if (features[5] < -0.027645f) {
                                          if (features[3] < 1.225893f) {
                                              return 1;
                                          } else {
                                              if (features[13] < -0.130782f) {
                                                  return 1;
                                              } else {
                                                  return 0;
                                              }
                                          }
                                      } else {
                                          if (features[15] < -0.116823f) {
                                              if (features[2] < -0.583052f) {
                                                  return 1;
                                              } else {
                                                  return 0;
                                              }
                                          } else {
                                              if (features[14] < 0.141648f) {
                                                  if (features[5] < 0.016598f) {
                                                      return 0;
                                                  } else {
                                                      if (features[12] < 0.225759f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              } else {
                                                  return 1;
                                              }
                                          }
                                      }
                                  }
                              } else {
                                  if (features[5] < -0.031171f) {
                                      if (features[15] < -0.249482f) {
                                          return 1;
                                      } else {
                                          return 0;
                                      }
                                  } else {
                                      if (features[15] < -0.166333f) {
                                          return 1;
                                      } else {
                                          if (features[4] < 2.076339f) {
                                              if (features[14] < -0.024484f) {
                                                  return 1;
                                              } else {
                                                  return 0;
                                              }
                                          } else {
                                              return 0;
                                          }
                                      }
                                  }
                              }
                          } else {
                              if (features[15] < -0.263258f) {
                                  return 1;
                              } else {
                                  if (features[13] < -0.147657f) {
                                      if (features[12] < 0.058467f) {
                                          if (features[14] < -0.374869f) {
                                              if (features[15] < 0.064424f) {
                                                  return 0;
                                              } else {
                                                  return 3;
                                              }
                                          } else {
                                              if (features[14] < -0.136065f) {
                                                  if (features[4] < 2.726638f) {
                                                      if (features[5] < -0.019650f) {
                                                          return 3;
                                                      } else {
                                                          return 3;
                                                      }
                                                  } else {
                                                      return 0;
                                                  }
                                              } else {
                                                  return 0;
                                              }
                                          }
                                      } else {
                                          return 0;
                                      }
                                  } else {
                                      if (features[14] < 0.172126f) {
                                          if (features[4] < 2.521802f) {
                                              if (features[9] < -0.589118f) {
                                                  return 3;
                                              } else {
                                                  if (features[3] < 1.880118f) {
                                                      if (features[15] < -0.143713f) {
                                                          return 3;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      if (features[0] < -0.014294f) {
                                                          return 3;
                                                      } else {
                                                          return 3;
                                                      }
                                                  }
                                              }
                                          } else {
                                              if (features[9] < -0.535248f) {
                                                  if (features[9] < -0.546529f) {
                                                      return 0;
                                                  } else {
                                                      if (features[8] < 0.700666f) {
                                                          return 3;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              } else {
                                                  return 0;
                                              }
                                          }
                                      } else {
                                          if (features[10] < 0.096380f) {
                                              return 3;
                                          } else {
                                              return 1;
                                          }
                                      }
                                  }
                              }
                          }
                      }
                  } else {
                      if (features[4] < 2.165103f) {
                          if (features[1] < 0.221308f) {
                              if (features[3] < 0.371740f) {
                                  if (features[12] < 0.109387f) {
                                      if (features[8] < 0.053474f) {
                                          return 0;
                                      } else {
                                          return 0;
                                      }
                                  } else {
                                      return 3;
                                  }
                              } else {
                                  if (features[15] < -0.031577f) {
                                      return 2;
                                  } else {
                                      return 2;
                                  }
                              }
                          } else {
                              if (features[0] < 0.044746f) {
                                  return 0;
                              } else {
                                  return 0;
                              }
                          }
                      } else {
                          if (features[12] < -0.051886f) {
                              return 3;
                          } else {
                              return 3;
                          }
                      }
                  }
              }
          } else {
              if (features[12] < -0.140296f) {
                  if (features[12] < -0.154660f) {
                      if (features[15] < -0.755822f) {
                          return 1;
                      } else {
                          if (features[13] < 0.056257f) {
                              if (features[8] < 0.378860f) {
                                  if (features[15] < -0.168880f) {
                                      if (features[5] < 0.207455f) {
                                          if (features[1] < 0.514423f) {
                                              if (features[0] < -0.011843f) {
                                                  return 2;
                                              } else {
                                                  if (features[11] < 0.817130f) {
                                                      if (features[4] < 2.019948f) {
                                                          return 2;
                                                      } else {
                                                          return 2;
                                                      }
                                                  } else {
                                                      if (features[11] < 0.847222f) {
                                                          return 2;
                                                      } else {
                                                          return 2;
                                                      }
                                                  }
                                              }
                                          } else {
                                              if (features[15] < -0.493289f) {
                                                  if (features[2] < -1.599049f) {
                                                      return 2;
                                                  } else {
                                                      return 2;
                                                  }
                                              } else {
                                                  if (features[7] < 0.537002f) {
                                                      return 2;
                                                  } else {
                                                      return 2;
                                                  }
                                              }
                                          }
                                      } else {
                                          if (features[7] < 0.410388f) {
                                              return 2;
                                          } else {
                                              if (features[12] < -0.497103f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          }
                                      }
                                  } else {
                                      if (features[1] < 0.538759f) {
                                          if (features[4] < 2.159643f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      } else {
                                          return 2;
                                      }
                                  }
                              } else {
                                  return 3;
                              }
                          } else {
                              if (features[13] < 0.558960f) {
                                  if (features[13] < 0.093562f) {
                                      return 2;
                                  } else {
                                      if (features[9] < -0.732710f) {
                                          if (features[13] < 0.252449f) {
                                              if (features[11] < 0.497685f) {
                                                  return 2;
                                              } else {
                                                  if (features[12] < -0.436221f) {
                                                      return 2;
                                                  } else {
                                                      if (features[11] < 0.530093f) {
                                                          return 2;
                                                      } else {
                                                          return 2;
                                                      }
                                                  }
                                              }
                                          } else {
                                              if (features[14] < 0.941636f) {
                                                  if (features[1] < 0.539346f) {
                                                      if (features[10] < 0.535706f) {
                                                          return 2;
                                                      } else {
                                                          return 2;
                                                      }
                                                  } else {
                                                      return 2;
                                                  }
                                              } else {
                                                  if (features[9] < -1.055578f) {
                                                      return 2;
                                                  } else {
                                                      return 2;
                                                  }
                                              }
                                          }
                                      } else {
                                          if (features[0] < -0.049415f) {
                                              return 2;
                                          } else {
                                              if (features[6] < 0.179018f) {
                                                  if (features[15] < -0.439776f) {
                                                      return 2;
                                                  } else {
                                                      return 2;
                                                  }
                                              } else {
                                                  if (features[7] < 0.323138f) {
                                                      return 2;
                                                  } else {
                                                      if (features[9] < -0.621605f) {
                                                          return 2;
                                                      } else {
                                                          return 2;
                                                      }
                                                  }
                                              }
                                          }
                                      }
                                  }
                              } else {
                                  return 2;
                              }
                          }
                      }
                  } else {
                      if (features[3] < 1.293562f) {
                          return 2;
                      } else {
                          return 3;
                      }
                  }
              } else {
                  if (features[0] < 0.009662f) {
                      if (features[13] < 0.073553f) {
                          if (features[3] < 1.303666f) {
                              if (features[0] < -0.101089f) {
                                  return 2;
                              } else {
                                  if (features[14] < -0.150156f) {
                                      return 2;
                                  } else {
                                      if (features[9] < -0.614846f) {
                                          return 2;
                                      } else {
                                          return 2;
                                      }
                                  }
                              }
                          } else {
                              if (features[6] < 0.389298f) {
                                  if (features[1] < 0.584908f) {
                                      if (features[7] < 0.420043f) {
                                          if (features[11] < 0.511574f) {
                                              return 0;
                                          } else {
                                              if (features[3] < 1.469830f) {
                                                  return 3;
                                              } else {
                                                  if (features[8] < 0.466112f) {
                                                      return 3;
                                                  } else {
                                                      return 3;
                                                  }
                                              }
                                          }
                                      } else {
                                          return 3;
                                      }
                                  } else {
                                      return 3;
                                  }
                              } else {
                                  if (features[3] < 2.231192f) {
                                      return 2;
                                  } else {
                                      return 0;
                                  }
                              }
                          }
                      } else {
                          if (features[8] < 0.303018f) {
                              if (features[0] < -0.052809f) {
                                  if (features[4] < 2.427066f) {
                                      if (features[3] < 1.070904f) {
                                          return 2;
                                      } else {
                                          return 2;
                                      }
                                  } else {
                                      return 2;
                                  }
                              } else {
                                  return 0;
                              }
                          } else {
                              return 3;
                          }
                      }
                  } else {
                      if (features[10] < 0.482879f) {
                          if (features[6] < 0.173425f) {
                              if (features[15] < -0.325908f) {
                                  return 0;
                              } else {
                                  return 0;
                              }
                          } else {
                              if (features[3] < 2.126864f) {
                                  return 1;
                              } else {
                                  return 0;
                              }
                          }
                      } else {
                          if (features[10] < 0.660444f) {
                              if (features[9] < -0.179568f) {
                                  if (features[7] < 0.343920f) {
                                      if (features[3] < 0.766913f) {
                                          if (features[0] < 0.088734f) {
                                              if (features[15] < -0.864638f) {
                                                  return 1;
                                              } else {
                                                  return 1;
                                              }
                                          } else {
                                              return 1;
                                          }
                                      } else {
                                          return 1;
                                      }
                                  } else {
                                      return 1;
                                  }
                              } else {
                                  if (features[7] < 0.294415f) {
                                      return 0;
                                  } else {
                                      return 3;
                                  }
                              }
                          } else {
                              return 2;
                          }
                      }
                  }
              }
          }
        }
        

static inline int32_t random_forest_tree_2(const float *features, int32_t features_length) {
          if (features[12] < -0.106787f) {
              if (features[2] < -0.911590f) {
                  if (features[10] < 0.334953f) {
                      if (features[15] < -0.289593f) {
                          if (features[13] < 0.218818f) {
                              if (features[7] < 0.248519f) {
                                  if (features[4] < 3.054042f) {
                                      return 1;
                                  } else {
                                      return 1;
                                  }
                              } else {
                                  return 1;
                              }
                          } else {
                              return 1;
                          }
                      } else {
                          if (features[9] < -0.442126f) {
                              if (features[14] < 0.261883f) {
                                  return 2;
                              } else {
                                  if (features[5] < -0.070931f) {
                                      return 2;
                                  } else {
                                      return 2;
                                  }
                              }
                          } else {
                              if (features[7] < 0.184411f) {
                                  return 0;
                              } else {
                                  return 1;
                              }
                          }
                      }
                  } else {
                      if (features[15] < -0.739716f) {
                          return 1;
                      } else {
                          if (features[14] < 0.221741f) {
                              if (features[8] < 0.276603f) {
                                  if (features[11] < 0.520833f) {
                                      if (features[10] < 0.553116f) {
                                          return 2;
                                      } else {
                                          if (features[3] < 1.192664f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      }
                                  } else {
                                      if (features[6] < 0.155329f) {
                                          return 2;
                                      } else {
                                          if (features[3] < 0.981286f) {
                                              if (features[2] < -1.135361f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          } else {
                                              return 2;
                                          }
                                      }
                                  }
                              } else {
                                  if (features[12] < -0.222134f) {
                                      if (features[8] < 0.362716f) {
                                          if (features[2] < -1.569153f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      } else {
                                          return 2;
                                      }
                                  } else {
                                      if (features[12] < -0.139442f) {
                                          return 3;
                                      } else {
                                          return 2;
                                      }
                                  }
                              }
                          } else {
                              if (features[5] < 0.310478f) {
                                  if (features[8] < 0.296015f) {
                                      if (features[0] < 0.134507f) {
                                          if (features[13] < 0.129831f) {
                                              if (features[6] < 0.249329f) {
                                                  return 2;
                                              } else {
                                                  if (features[11] < 0.511574f) {
                                                      return 2;
                                                  } else {
                                                      if (features[8] < 0.178384f) {
                                                          return 2;
                                                      } else {
                                                          return 2;
                                                      }
                                                  }
                                              }
                                          } else {
                                              if (features[3] < 1.431489f) {
                                                  if (features[10] < 0.714288f) {
                                                      if (features[13] < 0.509767f) {
                                                          return 2;
                                                      } else {
                                                          return 2;
                                                      }
                                                  } else {
                                                      if (features[3] < 1.193610f) {
                                                          return 2;
                                                      } else {
                                                          return 2;
                                                      }
                                                  }
                                              } else {
                                                  return 2;
                                              }
                                          }
                                      } else {
                                          if (features[11] < 0.787037f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      }
                                  } else {
                                      if (features[10] < 0.375123f) {
                                          return 2;
                                      } else {
                                          return 2;
                                      }
                                  }
                              } else {
                                  return 2;
                              }
                          }
                      }
                  }
              } else {
                  if (features[11] < 0.525463f) {
                      if (features[3] < 1.499650f) {
                          if (features[1] < 0.286197f) {
                              if (features[9] < -0.402664f) {
                                  if (features[7] < 0.202625f) {
                                      return 0;
                                  } else {
                                      return 2;
                                  }
                              } else {
                                  if (features[8] < 0.357075f) {
                                      if (features[12] < -0.502003f) {
                                          if (features[3] < 0.600671f) {
                                              if (features[6] < 0.023783f) {
                                                  return 1;
                                              } else {
                                                  if (features[15] < -0.721281f) {
                                                      return 1;
                                                  } else {
                                                      return 1;
                                                  }
                                              }
                                          } else {
                                              if (features[7] < 0.153745f) {
                                                  return 1;
                                              } else {
                                                  return 1;
                                              }
                                          }
                                      } else {
                                          if (features[3] < 0.780802f) {
                                              if (features[1] < 0.206219f) {
                                                  if (features[11] < 0.516204f) {
                                                      if (features[15] < -0.292435f) {
                                                          return 1;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      return 1;
                                                  }
                                              } else {
                                                  if (features[5] < -0.026903f) {
                                                      if (features[9] < -0.301900f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      return 2;
                                                  }
                                              }
                                          } else {
                                              if (features[7] < 0.135117f) {
                                                  return 0;
                                              } else {
                                                  if (features[9] < -0.233482f) {
                                                      if (features[12] < -0.121627f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      return 1;
                                                  }
                                              }
                                          }
                                      }
                                  } else {
                                      if (features[12] < -0.145097f) {
                                          if (features[0] < 0.010794f) {
                                              if (features[15] < -0.219292f) {
                                                  return 1;
                                              } else {
                                                  return 1;
                                              }
                                          } else {
                                              return 1;
                                          }
                                      } else {
                                          return 1;
                                      }
                                  }
                              }
                          } else {
                              if (features[8] < 0.345713f) {
                                  if (features[0] < -0.058690f) {
                                      if (features[3] < 1.271241f) {
                                          if (features[14] < 0.556591f) {
                                              if (features[6] < 0.187461f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          } else {
                                              if (features[10] < 0.276707f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          }
                                      } else {
                                          return 2;
                                      }
                                  } else {
                                      if (features[10] < 0.410015f) {
                                          if (features[7] < 0.246710f) {
                                              if (features[4] < 1.501415f) {
                                                  return 2;
                                              } else {
                                                  if (features[5] < 0.016554f) {
                                                      if (features[14] < 0.320472f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      return 0;
                                                  }
                                              }
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          if (features[9] < -0.480949f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      }
                                  }
                              } else {
                                  return 1;
                              }
                          }
                      } else {
                          if (features[14] < -0.098479f) {
                              return 3;
                          } else {
                              if (features[9] < -0.268549f) {
                                  if (features[14] < 0.031098f) {
                                      return 0;
                                  } else {
                                      if (features[5] < -0.038020f) {
                                          return 3;
                                      } else {
                                          if (features[5] < -0.018264f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      }
                                  }
                              } else {
                                  return 1;
                              }
                          }
                      }
                  } else {
                      if (features[4] < 2.196661f) {
                          if (features[4] < 0.859891f) {
                              return 2;
                          } else {
                              if (features[8] < 0.187885f) {
                                  if (features[14] < -0.140220f) {
                                      return 2;
                                  } else {
                                      if (features[2] < -0.802011f) {
                                          return 2;
                                      } else {
                                          return 2;
                                      }
                                  }
                              } else {
                                  return 2;
                              }
                          }
                      } else {
                          return 3;
                      }
                  }
              }
          } else {
              if (features[11] < 0.525463f) {
                  if (features[10] < 0.416618f) {
                      if (features[2] < -0.543225f) {
                          if (features[5] < -0.094880f) {
                              if (features[3] < 1.410668f) {
                                  if (features[15] < -0.257297f) {
                                      return 1;
                                  } else {
                                      return 0;
                                  }
                              } else {
                                  if (features[3] < 1.530689f) {
                                      return 0;
                                  } else {
                                      return 3;
                                  }
                              }
                          } else {
                              if (features[14] < 0.201924f) {
                                  if (features[15] < -0.231459f) {
                                      if (features[8] < 0.173343f) {
                                          return 2;
                                      } else {
                                          if (features[6] < 0.138405f) {
                                              if (features[6] < 0.078110f) {
                                                  return 1;
                                              } else {
                                                  return 1;
                                              }
                                          } else {
                                              return 1;
                                          }
                                      }
                                  } else {
                                      if (features[15] < -0.136254f) {
                                          if (features[6] < 0.201477f) {
                                              if (features[10] < 0.170606f) {
                                                  if (features[0] < 0.000232f) {
                                                      return 0;
                                                  } else {
                                                      return 1;
                                                  }
                                              } else {
                                                  if (features[3] < 0.787960f) {
                                                      return 0;
                                                  } else {
                                                      return 0;
                                                  }
                                              }
                                          } else {
                                              return 3;
                                          }
                                      } else {
                                          if (features[6] < 0.202534f) {
                                              if (features[0] < -0.061113f) {
                                                  if (features[10] < 0.171331f) {
                                                      return 0;
                                                  } else {
                                                      return 0;
                                                  }
                                              } else {
                                                  if (features[13] < 0.432197f) {
                                                      if (features[15] < -0.076076f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      return 0;
                                                  }
                                              }
                                          } else {
                                              if (features[2] < -0.642371f) {
                                                  if (features[4] < 2.521317f) {
                                                      if (features[15] < -0.009722f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      if (features[9] < -0.593981f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              } else {
                                                  if (features[0] < -0.040762f) {
                                                      return 0;
                                                  } else {
                                                      return 3;
                                                  }
                                              }
                                          }
                                      }
                                  }
                              } else {
                                  if (features[8] < 0.386658f) {
                                      if (features[10] < 0.346187f) {
                                          if (features[1] < 0.283911f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          return 2;
                                      }
                                  } else {
                                      return 1;
                                  }
                              }
                          }
                      } else {
                          if (features[3] < 1.777670f) {
                              if (features[14] < 0.259933f) {
                                  if (features[15] < -0.350093f) {
                                      return 1;
                                  } else {
                                      if (features[7] < 0.172688f) {
                                          if (features[10] < 0.070552f) {
                                              if (features[4] < 1.001840f) {
                                                  return 0;
                                              } else {
                                                  if (features[3] < 0.621196f) {
                                                      return 0;
                                                  } else {
                                                      return 0;
                                                  }
                                              }
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          if (features[5] < -0.083731f) {
                                              return 3;
                                          } else {
                                              if (features[2] < -0.285309f) {
                                                  if (features[0] < -0.047882f) {
                                                      if (features[8] < 0.403285f) {
                                                          return 2;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      if (features[3] < 1.719211f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              } else {
                                                  return 0;
                                              }
                                          }
                                      }
                                  }
                              } else {
                                  if (features[15] < 0.016622f) {
                                      if (features[7] < 0.118950f) {
                                          return 0;
                                      } else {
                                          return 1;
                                      }
                                  } else {
                                      return 0;
                                  }
                              }
                          } else {
                              if (features[3] < 1.835224f) {
                                  if (features[8] < 0.463144f) {
                                      return 3;
                                  } else {
                                      return 0;
                                  }
                              } else {
                                  if (features[12] < 0.009588f) {
                                      if (features[1] < 0.384197f) {
                                          return 3;
                                      } else {
                                          return 3;
                                      }
                                  } else {
                                      return 3;
                                  }
                              }
                          }
                      }
                  } else {
                      if (features[14] < -0.173629f) {
                          if (features[4] < 2.292875f) {
                              return 0;
                          } else {
                              if (features[11] < 0.511574f) {
                                  return 3;
                              } else {
                                  if (features[1] < 0.551896f) {
                                      return 3;
                                  } else {
                                      return 3;
                                  }
                              }
                          }
                      } else {
                          if (features[8] < 0.329661f) {
                              if (features[12] < 0.005667f) {
                                  if (features[9] < -0.558685f) {
                                      if (features[3] < 1.561969f) {
                                          if (features[6] < 0.240463f) {
                                              return 2;
                                          } else {
                                              if (features[3] < 1.201563f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          }
                                      } else {
                                          return 0;
                                      }
                                  } else {
                                      return 2;
                                  }
                              } else {
                                  if (features[0] < -0.061883f) {
                                      return 2;
                                  } else {
                                      return 0;
                                  }
                              }
                          } else {
                              if (features[15] < -0.302121f) {
                                  return 1;
                              } else {
                                  if (features[4] < 2.872820f) {
                                      if (features[2] < -0.928596f) {
                                          return 3;
                                      } else {
                                          if (features[9] < -0.544049f) {
                                              return 3;
                                          } else {
                                              return 0;
                                          }
                                      }
                                  } else {
                                      return 0;
                                  }
                              }
                          }
                      }
                  }
              } else {
                  if (features[11] < 0.562500f) {
                      if (features[8] < 0.317173f) {
                          if (features[0] < -0.075135f) {
                              return 2;
                          } else {
                              return 2;
                          }
                      } else {
                          if (features[6] < 0.313449f) {
                              if (features[0] < -0.008845f) {
                                  return 3;
                              } else {
                                  return 3;
                              }
                          } else {
                              return 3;
                          }
                      }
                  } else {
                      if (features[15] < -0.662794f) {
                          return 1;
                      } else {
                          if (features[12] < 0.254532f) {
                              if (features[1] < 0.470538f) {
                                  if (features[14] < 0.114290f) {
                                      if (features[13] < -0.197837f) {
                                          return 2;
                                      } else {
                                          if (features[9] < -0.264495f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      }
                                  } else {
                                      return 2;
                                  }
                              } else {
                                  return 3;
                              }
                          } else {
                              return 3;
                          }
                      }
                  }
              }
          }
        }
        

static inline int32_t random_forest_tree_3(const float *features, int32_t features_length) {
          if (features[7] < 0.334058f) {
              if (features[14] < 0.196036f) {
                  if (features[2] < -0.540428f) {
                      if (features[7] < 0.293375f) {
                          if (features[1] < 0.370885f) {
                              if (features[5] < -0.067320f) {
                                  if (features[3] < 0.900454f) {
                                      if (features[1] < 0.276009f) {
                                          return 0;
                                      } else {
                                          return 0;
                                      }
                                  } else {
                                      if (features[9] < -0.392893f) {
                                          return 0;
                                      } else {
                                          if (features[2] < -0.714818f) {
                                              if (features[6] < 0.100287f) {
                                                  if (features[6] < 0.081535f) {
                                                      return 1;
                                                  } else {
                                                      return 1;
                                                  }
                                              } else {
                                                  return 1;
                                              }
                                          } else {
                                              return 1;
                                          }
                                      }
                                  }
                              } else {
                                  if (features[0] < -0.052059f) {
                                      if (features[10] < 0.260357f) {
                                          return 0;
                                      } else {
                                          if (features[3] < 0.676022f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      }
                                  } else {
                                      if (features[13] < -0.107800f) {
                                          if (features[12] < -0.046518f) {
                                              if (features[5] < -0.004341f) {
                                                  if (features[8] < 0.371087f) {
                                                      if (features[7] < 0.168510f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      if (features[12] < -0.173800f) {
                                                          return 1;
                                                      } else {
                                                          return 1;
                                                      }
                                                  }
                                              } else {
                                                  if (features[0] < 0.056257f) {
                                                      if (features[10] < 0.279345f) {
                                                          return 0;
                                                      } else {
                                                          return 2;
                                                      }
                                                  } else {
                                                      if (features[4] < 1.592211f) {
                                                          return 2;
                                                      } else {
                                                          return 1;
                                                      }
                                                  }
                                              }
                                          } else {
                                              if (features[12] < 0.000490f) {
                                                  if (features[14] < -0.287224f) {
                                                      return 0;
                                                  } else {
                                                      if (features[15] < -0.041233f) {
                                                          return 1;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              } else {
                                                  if (features[11] < 0.520833f) {
                                                      return 0;
                                                  } else {
                                                      return 0;
                                                  }
                                              }
                                          }
                                      } else {
                                          if (features[11] < 0.726852f) {
                                              if (features[15] < -0.306653f) {
                                                  if (features[4] < 1.687923f) {
                                                      return 0;
                                                  } else {
                                                      if (features[2] < -0.813901f) {
                                                          return 1;
                                                      } else {
                                                          return 1;
                                                      }
                                                  }
                                              } else {
                                                  if (features[12] < 0.519147f) {
                                                      if (features[14] < 0.142038f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      return 0;
                                                  }
                                              }
                                          } else {
                                              if (features[11] < 0.814815f) {
                                                  return 1;
                                              } else {
                                                  if (features[6] < 0.049947f) {
                                                      return 2;
                                                  } else {
                                                      return 0;
                                                  }
                                              }
                                          }
                                      }
                                  }
                              }
                          } else {
                              if (features[15] < -0.253082f) {
                                  if (features[8] < 0.445642f) {
                                      return 2;
                                  } else {
                                      return 1;
                                  }
                              } else {
                                  if (features[7] < 0.271174f) {
                                      if (features[9] < -0.498719f) {
                                          if (features[5] < -0.066899f) {
                                              return 3;
                                          } else {
                                              if (features[7] < 0.229734f) {
                                                  return 0;
                                              } else {
                                                  return 0;
                                              }
                                          }
                                      } else {
                                          if (features[2] < -1.462391f) {
                                              if (features[15] < -0.136946f) {
                                                  return 0;
                                              } else {
                                                  return 0;
                                              }
                                          } else {
                                              if (features[15] < -0.099803f) {
                                                  if (features[15] < -0.102350f) {
                                                      return 0;
                                                  } else {
                                                      return 3;
                                                  }
                                              } else {
                                                  return 0;
                                              }
                                          }
                                      }
                                  } else {
                                      if (features[0] < -0.044318f) {
                                          return 0;
                                      } else {
                                          if (features[13] < -0.098681f) {
                                              if (features[2] < -0.704412f) {
                                                  return 0;
                                              } else {
                                                  return 3;
                                              }
                                          } else {
                                              if (features[2] < -0.625170f) {
                                                  return 0;
                                              } else {
                                                  return 3;
                                              }
                                          }
                                      }
                                  }
                              }
                          }
                      } else {
                          if (features[0] < 0.035417f) {
                              if (features[11] < 0.520833f) {
                                  if (features[13] < -0.311761f) {
                                      if (features[2] < -0.771760f) {
                                          if (features[14] < -0.535114f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          return 3;
                                      }
                                  } else {
                                      if (features[15] < -0.088420f) {
                                          return 2;
                                      } else {
                                          if (features[1] < 0.464882f) {
                                              if (features[7] < 0.303890f) {
                                                  return 0;
                                              } else {
                                                  return 0;
                                              }
                                          } else {
                                              if (features[12] < 0.321063f) {
                                                  return 0;
                                              } else {
                                                  return 0;
                                              }
                                          }
                                      }
                                  }
                              } else {
                                  if (features[1] < 0.460101f) {
                                      return 3;
                                  } else {
                                      if (features[13] < -0.035719f) {
                                          return 3;
                                      } else {
                                          return 3;
                                      }
                                  }
                              }
                          } else {
                              if (features[13] < -0.375354f) {
                                  return 2;
                              } else {
                                  if (features[15] < -0.376792f) {
                                      if (features[15] < -0.639511f) {
                                          return 1;
                                      } else {
                                          return 1;
                                      }
                                  } else {
                                      return 0;
                                  }
                              }
                          }
                      }
                  } else {
                      if (features[1] < 0.326247f) {
                          if (features[12] < -0.318327f) {
                              if (features[15] < -0.424412f) {
                                  if (features[1] < 0.208250f) {
                                      return 1;
                                  } else {
                                      return 0;
                                  }
                              } else {
                                  return 2;
                              }
                          } else {
                              if (features[11] < 0.891204f) {
                                  if (features[15] < -0.348275f) {
                                      if (features[2] < -0.312911f) {
                                          return 1;
                                      } else {
                                          return 0;
                                      }
                                  } else {
                                      if (features[0] < -0.045788f) {
                                          return 2;
                                      } else {
                                          if (features[4] < 0.458514f) {
                                              return 3;
                                          } else {
                                              if (features[15] < -0.080149f) {
                                                  if (features[1] < 0.306777f) {
                                                      if (features[6] < 0.089863f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      return 3;
                                                  }
                                              } else {
                                                  if (features[12] < 0.563959f) {
                                                      if (features[13] < 0.087979f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      if (features[10] < 0.124311f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              }
                                          }
                                      }
                                  }
                              } else {
                                  return 2;
                              }
                          }
                      } else {
                          if (features[0] < -0.019728f) {
                              if (features[4] < 2.318545f) {
                                  if (features[5] < -0.078531f) {
                                      return 3;
                                  } else {
                                      if (features[10] < 0.212204f) {
                                          return 0;
                                      } else {
                                          return 3;
                                      }
                                  }
                              } else {
                                  return 3;
                              }
                          } else {
                              if (features[8] < 0.347529f) {
                                  return 0;
                              } else {
                                  if (features[13] < -0.081467f) {
                                      if (features[11] < 0.520833f) {
                                          if (features[14] < -0.199978f) {
                                              if (features[7] < 0.202998f) {
                                                  return 3;
                                              } else {
                                                  return 3;
                                              }
                                          } else {
                                              return 3;
                                          }
                                      } else {
                                          return 3;
                                      }
                                  } else {
                                      if (features[5] < -0.029289f) {
                                          return 3;
                                      } else {
                                          if (features[4] < 2.082044f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      }
                                  }
                              }
                          }
                      }
                  }
              } else {
                  if (features[5] < 0.003484f) {
                      if (features[8] < 0.284906f) {
                          if (features[15] < -0.499351f) {
                              if (features[1] < 0.323115f) {
                                  if (features[9] < -0.267608f) {
                                      return 1;
                                  } else {
                                      if (features[8] < 0.129023f) {
                                          return 1;
                                      } else {
                                          if (features[14] < 0.641354f) {
                                              return 1;
                                          } else {
                                              return 1;
                                          }
                                      }
                                  }
                              } else {
                                  return 2;
                              }
                          } else {
                              if (features[3] < 0.706555f) {
                                  if (features[2] < -0.494290f) {
                                      if (features[10] < 0.247902f) {
                                          return 0;
                                      } else {
                                          if (features[5] < -0.003770f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      }
                                  } else {
                                      return 1;
                                  }
                              } else {
                                  if (features[5] < -0.045986f) {
                                      if (features[9] < -0.402420f) {
                                          if (features[12] < -0.266269f) {
                                              return 2;
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          return 1;
                                      }
                                  } else {
                                      if (features[9] < -0.488316f) {
                                          if (features[5] < -0.036319f) {
                                              return 2;
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          if (features[13] < -0.039351f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      }
                                  }
                              }
                          }
                      } else {
                          if (features[10] < 0.102931f) {
                              return 3;
                          } else {
                              if (features[9] < -0.422600f) {
                                  return 1;
                              } else {
                                  if (features[15] < -0.183731f) {
                                      if (features[4] < 1.821491f) {
                                          return 1;
                                      } else {
                                          if (features[7] < 0.208159f) {
                                              return 1;
                                          } else {
                                              return 1;
                                          }
                                      }
                                  } else {
                                      if (features[14] < 0.282841f) {
                                          return 1;
                                      } else {
                                          return 1;
                                      }
                                  }
                              }
                          }
                      }
                  } else {
                      if (features[10] < 0.287769f) {
                          if (features[9] < -0.369356f) {
                              if (features[15] < -0.111789f) {
                                  if (features[4] < 1.637714f) {
                                      if (features[8] < 0.087475f) {
                                          return 2;
                                      } else {
                                          return 2;
                                      }
                                  } else {
                                      return 0;
                                  }
                              } else {
                                  return 0;
                              }
                          } else {
                              if (features[11] < 0.685185f) {
                                  if (features[8] < 0.117827f) {
                                      return 1;
                                  } else {
                                      if (features[8] < 0.314427f) {
                                          return 1;
                                      } else {
                                          return 1;
                                      }
                                  }
                              } else {
                                  return 2;
                              }
                          }
                      } else {
                          if (features[0] < 0.116354f) {
                              if (features[11] < 0.511574f) {
                                  if (features[2] < -0.742740f) {
                                      return 2;
                                  } else {
                                      return 1;
                                  }
                              } else {
                                  if (features[3] < 0.919666f) {
                                      if (features[10] < 0.349845f) {
                                          if (features[15] < -0.339471f) {
                                              return 2;
                                          } else {
                                              if (features[2] < -0.578257f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          }
                                      } else {
                                          return 2;
                                      }
                                  } else {
                                      return 2;
                                  }
                              }
                          } else {
                              return 1;
                          }
                      }
                  }
              }
          } else {
              if (features[12] < -0.149224f) {
                  if (features[13] < -0.090126f) {
                      if (features[5] < -0.025415f) {
                          if (features[11] < 0.530093f) {
                              if (features[10] < 0.497543f) {
                                  return 2;
                              } else {
                                  if (features[8] < 0.253418f) {
                                      if (features[5] < -0.125133f) {
                                          return 2;
                                      } else {
                                          if (features[10] < 0.646028f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      }
                                  } else {
                                      return 2;
                                  }
                              }
                          } else {
                              return 3;
                          }
                      } else {
                          if (features[11] < 0.518519f) {
                              if (features[12] < -0.430719f) {
                                  if (features[11] < 0.506944f) {
                                      return 2;
                                  } else {
                                      return 2;
                                  }
                              } else {
                                  return 0;
                              }
                          } else {
                              if (features[1] < 0.739056f) {
                                  return 2;
                              } else {
                                  return 2;
                              }
                          }
                      }
                  } else {
                      if (features[6] < 0.265147f) {
                          if (features[5] < -0.067949f) {
                              if (features[9] < -0.725473f) {
                                  return 2;
                              } else {
                                  return 2;
                              }
                          } else {
                              if (features[2] < -1.044479f) {
                                  if (features[3] < 0.906015f) {
                                      if (features[9] < -0.565401f) {
                                          return 2;
                                      } else {
                                          return 2;
                                      }
                                  } else {
                                      return 2;
                                  }
                              } else {
                                  if (features[8] < 0.178620f) {
                                      return 2;
                                  } else {
                                      return 2;
                                  }
                              }
                          }
                      } else {
                          if (features[1] < 0.823514f) {
                              if (features[13] < 0.094794f) {
                                  if (features[8] < 0.198414f) {
                                      return 2;
                                  } else {
                                      if (features[10] < 0.670556f) {
                                          return 2;
                                      } else {
                                          return 2;
                                      }
                                  }
                              } else {
                                  if (features[4] < 2.512083f) {
                                      if (features[11] < 0.442130f) {
                                          return 2;
                                      } else {
                                          if (features[11] < 0.511574f) {
                                              if (features[7] < 0.447272f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          } else {
                                              if (features[11] < 0.530093f) {
                                                  return 2;
                                              } else {
                                                  if (features[0] < -0.101351f) {
                                                      return 2;
                                                  } else {
                                                      return 2;
                                                  }
                                              }
                                          }
                                      }
                                  } else {
                                      if (features[2] < -1.101321f) {
                                          if (features[8] < 0.206338f) {
                                              return 2;
                                          } else {
                                              if (features[5] < 0.177025f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          }
                                      } else {
                                          return 2;
                                      }
                                  }
                              }
                          } else {
                              return 2;
                          }
                      }
                  }
              } else {
                  if (features[3] < 1.328222f) {
                      if (features[14] < 0.071914f) {
                          if (features[5] < -0.091847f) {
                              if (features[15] < -0.072218f) {
                                  return 2;
                              } else {
                                  return 2;
                              }
                          } else {
                              if (features[7] < 0.360371f) {
                                  return 1;
                              } else {
                                  return 3;
                              }
                          }
                      } else {
                          if (features[11] < 0.511574f) {
                              return 2;
                          } else {
                              if (features[3] < 1.069408f) {
                                  return 2;
                              } else {
                                  return 2;
                              }
                          }
                      }
                  } else {
                      if (features[1] < 0.623286f) {
                          if (features[14] < 0.143508f) {
                              if (features[2] < -1.095729f) {
                                  return 1;
                              } else {
                                  if (features[5] < -0.021725f) {
                                      if (features[8] < 0.330919f) {
                                          return 3;
                                      } else {
                                          if (features[7] < 0.352866f) {
                                              return 3;
                                          } else {
                                              if (features[0] < -0.063587f) {
                                                  return 3;
                                              } else {
                                                  if (features[5] < -0.087259f) {
                                                      return 3;
                                                  } else {
                                                      return 3;
                                                  }
                                              }
                                          }
                                      }
                                  } else {
                                      if (features[1] < 0.538357f) {
                                          return 0;
                                      } else {
                                          return 3;
                                      }
                                  }
                              }
                          } else {
                              if (features[15] < -0.073640f) {
                                  return 2;
                              } else {
                                  return 2;
                              }
                          }
                      } else {
                          if (features[6] < 0.808842f) {
                              if (features[13] < -0.105336f) {
                                  return 2;
                              } else {
                                  if (features[12] < -0.047622f) {
                                      return 0;
                                  } else {
                                      return 0;
                                  }
                              }
                          } else {
                              return 2;
                          }
                      }
                  }
              }
          }
        }
        

static inline int32_t random_forest_tree_4(const float *features, int32_t features_length) {
          if (features[15] < -0.130701f) {
              if (features[10] < 0.281542f) {
                  if (features[7] < 0.268839f) {
                      if (features[8] < 0.280176f) {
                          if (features[12] < -0.549252f) {
                              if (features[13] < 0.224695f) {
                                  if (features[12] < -0.618114f) {
                                      if (features[10] < 0.175506f) {
                                          return 1;
                                      } else {
                                          if (features[15] < -0.741464f) {
                                              return 1;
                                          } else {
                                              return 1;
                                          }
                                      }
                                  } else {
                                      return 1;
                                  }
                              } else {
                                  return 1;
                              }
                          } else {
                              if (features[3] < 0.719692f) {
                                  if (features[10] < 0.239921f) {
                                      if (features[9] < -0.316580f) {
                                          return 2;
                                      } else {
                                          if (features[2] < -0.339169f) {
                                              if (features[8] < 0.168856f) {
                                                  if (features[14] < 0.063000f) {
                                                      return 0;
                                                  } else {
                                                      return 0;
                                                  }
                                              } else {
                                                  return 1;
                                              }
                                          } else {
                                              return 1;
                                          }
                                      }
                                  } else {
                                      if (features[9] < -0.342422f) {
                                          return 2;
                                      } else {
                                          return 2;
                                      }
                                  }
                              } else {
                                  if (features[14] < 0.641354f) {
                                      if (features[9] < -0.146539f) {
                                          if (features[1] < 0.242865f) {
                                              return 0;
                                          } else {
                                              if (features[9] < -0.392344f) {
                                                  return 0;
                                              } else {
                                                  return 0;
                                              }
                                          }
                                      } else {
                                          return 1;
                                      }
                                  } else {
                                      return 1;
                                  }
                              }
                          }
                      } else {
                          if (features[1] < 0.308337f) {
                              if (features[11] < 0.516204f) {
                                  return 1;
                              } else {
                                  if (features[0] < 0.007415f) {
                                      if (features[3] < 1.474220f) {
                                          if (features[12] < -0.203343f) {
                                              return 1;
                                          } else {
                                              if (features[12] < -0.024146f) {
                                                  if (features[8] < 0.424555f) {
                                                      return 1;
                                                  } else {
                                                      return 1;
                                                  }
                                              } else {
                                                  if (features[3] < 1.158629f) {
                                                      return 1;
                                                  } else {
                                                      return 1;
                                                  }
                                              }
                                          }
                                      } else {
                                          return 1;
                                      }
                                  } else {
                                      if (features[2] < -0.869515f) {
                                          return 0;
                                      } else {
                                          if (features[15] < -0.210244f) {
                                              return 1;
                                          } else {
                                              return 1;
                                          }
                                      }
                                  }
                              }
                          } else {
                              if (features[15] < -0.260161f) {
                                  if (features[8] < 0.666323f) {
                                      return 1;
                                  } else {
                                      if (features[10] < 0.171817f) {
                                          return 1;
                                      } else {
                                          return 1;
                                      }
                                  }
                              } else {
                                  if (features[11] < 0.520833f) {
                                      return 0;
                                  } else {
                                      if (features[9] < -0.189428f) {
                                          return 3;
                                      } else {
                                          return 0;
                                      }
                                  }
                              }
                          }
                      }
                  } else {
                      if (features[14] < 0.241852f) {
                          if (features[10] < 0.249574f) {
                              return 3;
                          } else {
                              return 2;
                          }
                      } else {
                          if (features[6] < 0.200958f) {
                              return 2;
                          } else {
                              return 2;
                          }
                      }
                  }
              } else {
                  if (features[14] < 0.155200f) {
                      if (features[12] < -0.132347f) {
                          if (features[6] < 0.233764f) {
                              if (features[13] < -0.442270f) {
                                  if (features[3] < 0.925274f) {
                                      if (features[5] < 0.009412f) {
                                          return 2;
                                      } else {
                                          if (features[14] < -0.058439f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      }
                                  } else {
                                      if (features[9] < -0.423058f) {
                                          return 2;
                                      } else {
                                          return 1;
                                      }
                                  }
                              } else {
                                  if (features[5] < -0.028932f) {
                                      return 3;
                                  } else {
                                      if (features[9] < -0.354313f) {
                                          return 2;
                                      } else {
                                          return 1;
                                      }
                                  }
                              }
                          } else {
                              if (features[12] < -0.196378f) {
                                  if (features[10] < 0.510665f) {
                                      return 2;
                                  } else {
                                      if (features[3] < 1.597943f) {
                                          if (features[6] < 0.329468f) {
                                              if (features[11] < 0.548611f) {
                                                  if (features[15] < -0.362731f) {
                                                      return 2;
                                                  } else {
                                                      return 2;
                                                  }
                                              } else {
                                                  return 2;
                                              }
                                          } else {
                                              if (features[13] < -0.505666f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          }
                                      } else {
                                          return 2;
                                      }
                                  }
                              } else {
                                  return 2;
                              }
                          }
                      } else {
                          if (features[8] < 0.114174f) {
                              if (features[11] < 0.562500f) {
                                  return 2;
                              } else {
                                  return 0;
                              }
                          } else {
                              if (features[9] < -0.185580f) {
                                  if (features[1] < 0.468377f) {
                                      if (features[13] < -0.112901f) {
                                          return 2;
                                      } else {
                                          if (features[11] < 0.516204f) {
                                              return 0;
                                          } else {
                                              if (features[2] < -1.180697f) {
                                                  return 1;
                                              } else {
                                                  if (features[15] < -0.356009f) {
                                                      if (features[5] < 0.007988f) {
                                                          return 1;
                                                      } else {
                                                          return 1;
                                                      }
                                                  } else {
                                                      return 0;
                                                  }
                                              }
                                          }
                                      }
                                  } else {
                                      return 3;
                                  }
                              } else {
                                  if (features[13] < -0.055930f) {
                                      return 0;
                                  } else {
                                      return 0;
                                  }
                              }
                          }
                      }
                  } else {
                      if (features[7] < 0.226423f) {
                          if (features[11] < 0.562500f) {
                              if (features[10] < 0.325263f) {
                                  if (features[15] < -0.404658f) {
                                      return 1;
                                  } else {
                                      return 1;
                                  }
                              } else {
                                  return 0;
                              }
                          } else {
                              if (features[4] < 1.164140f) {
                                  return 2;
                              } else {
                                  return 2;
                              }
                          }
                      } else {
                          if (features[6] < 0.240686f) {
                              if (features[13] < -0.009222f) {
                                  if (features[1] < 0.374431f) {
                                      return 2;
                                  } else {
                                      return 2;
                                  }
                              } else {
                                  if (features[2] < -0.977800f) {
                                      if (features[5] < 0.166053f) {
                                          if (features[3] < 0.563880f) {
                                              return 2;
                                          } else {
                                              if (features[3] < 1.040825f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          }
                                      } else {
                                          return 2;
                                      }
                                  } else {
                                      if (features[3] < 1.101936f) {
                                          if (features[15] < -0.383129f) {
                                              if (features[7] < 0.260878f) {
                                                  return 2;
                                              } else {
                                                  if (features[2] < -0.734665f) {
                                                      return 2;
                                                  } else {
                                                      return 2;
                                                  }
                                              }
                                          } else {
                                              return 2;
                                          }
                                      } else {
                                          if (features[13] < 0.263376f) {
                                              return 2;
                                          } else {
                                              return 1;
                                          }
                                      }
                                  }
                              }
                          } else {
                              if (features[3] < 2.660807f) {
                                  if (features[5] < 0.375314f) {
                                      if (features[5] < 0.231724f) {
                                          if (features[11] < 0.400463f) {
                                              if (features[2] < -1.431369f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          } else {
                                              if (features[14] < 0.371824f) {
                                                  if (features[1] < 0.502127f) {
                                                      return 2;
                                                  } else {
                                                      if (features[3] < 1.327066f) {
                                                          return 2;
                                                      } else {
                                                          return 2;
                                                      }
                                                  }
                                              } else {
                                                  if (features[12] < -0.151983f) {
                                                      if (features[6] < 0.252595f) {
                                                          return 2;
                                                      } else {
                                                          return 2;
                                                      }
                                                  } else {
                                                      return 2;
                                                  }
                                              }
                                          }
                                      } else {
                                          if (features[10] < 0.791851f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      }
                                  } else {
                                      return 2;
                                  }
                              } else {
                                  return 2;
                              }
                          }
                      }
                  }
              }
          } else {
              if (features[13] < -0.140400f) {
                  if (features[6] < 0.106388f) {
                      if (features[8] < 0.158221f) {
                          if (features[11] < 0.627315f) {
                              if (features[12] < -0.054603f) {
                                  return 3;
                              } else {
                                  return 0;
                              }
                          } else {
                              return 3;
                          }
                      } else {
                          if (features[8] < 0.434340f) {
                              return 0;
                          } else {
                              if (features[3] < 1.454007f) {
                                  if (features[7] < 0.143092f) {
                                      return 0;
                                  } else {
                                      return 1;
                                  }
                              } else {
                                  return 0;
                              }
                          }
                      }
                  } else {
                      if (features[14] < -0.428921f) {
                          if (features[10] < 0.446851f) {
                              return 0;
                          } else {
                              return 3;
                          }
                      } else {
                          if (features[0] < -0.098793f) {
                              return 2;
                          } else {
                              if (features[11] < 0.511574f) {
                                  if (features[7] < 0.318834f) {
                                      if (features[8] < 0.179273f) {
                                          return 0;
                                      } else {
                                          return 0;
                                      }
                                  } else {
                                      return 3;
                                  }
                              } else {
                                  if (features[9] < -0.199200f) {
                                      if (features[0] < -0.073743f) {
                                          return 3;
                                      } else {
                                          if (features[2] < -1.021655f) {
                                              if (features[6] < 0.371726f) {
                                                  return 0;
                                              } else {
                                                  return 0;
                                              }
                                          } else {
                                              if (features[4] < 2.124777f) {
                                                  return 3;
                                              } else {
                                                  if (features[14] < -0.097596f) {
                                                      if (features[0] < 0.022791f) {
                                                          return 3;
                                                      } else {
                                                          return 3;
                                                      }
                                                  } else {
                                                      return 3;
                                                  }
                                              }
                                          }
                                      }
                                  } else {
                                      return 0;
                                  }
                              }
                          }
                      }
                  }
              } else {
                  if (features[10] < 0.467708f) {
                      if (features[11] < 0.895833f) {
                          if (features[15] < -0.095627f) {
                              if (features[5] < -0.070450f) {
                                  if (features[4] < 2.126704f) {
                                      return 0;
                                  } else {
                                      return 3;
                                  }
                              } else {
                                  if (features[12] < -0.121251f) {
                                      if (features[13] < 0.034969f) {
                                          if (features[6] < 0.116035f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          return 1;
                                      }
                                  } else {
                                      if (features[9] < -0.467331f) {
                                          if (features[0] < -0.044415f) {
                                              return 0;
                                          } else {
                                              return 3;
                                          }
                                      } else {
                                          if (features[0] < -0.040991f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      }
                                  }
                              }
                          } else {
                              if (features[5] < -0.082783f) {
                                  if (features[3] < 1.553222f) {
                                      if (features[11] < 0.511574f) {
                                          return 0;
                                      } else {
                                          return 0;
                                      }
                                  } else {
                                      if (features[15] < 0.004232f) {
                                          return 3;
                                      } else {
                                          return 0;
                                      }
                                  }
                              } else {
                                  if (features[14] < 0.203654f) {
                                      if (features[7] < 0.067241f) {
                                          return 3;
                                      } else {
                                          if (features[13] < -0.073415f) {
                                              if (features[6] < 0.160906f) {
                                                  if (features[1] < 0.386287f) {
                                                      return 0;
                                                  } else {
                                                      if (features[1] < 0.390140f) {
                                                          return 3;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              } else {
                                                  if (features[3] < 1.870985f) {
                                                      return 0;
                                                  } else {
                                                      if (features[4] < 2.886712f) {
                                                          return 3;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              }
                                          } else {
                                              if (features[14] < 0.149583f) {
                                                  if (features[7] < 0.254864f) {
                                                      if (features[14] < -0.512354f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      if (features[2] < -0.561717f) {
                                                          return 0;
                                                      } else {
                                                          return 3;
                                                      }
                                                  }
                                              } else {
                                                  if (features[3] < 1.650846f) {
                                                      return 0;
                                                  } else {
                                                      return 0;
                                                  }
                                              }
                                          }
                                      }
                                  } else {
                                      if (features[8] < 0.370125f) {
                                          if (features[8] < 0.242463f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          return 1;
                                      }
                                  }
                              }
                          }
                      } else {
                          if (features[3] < 0.375028f) {
                              return 3;
                          } else {
                              if (features[7] < 0.147463f) {
                                  return 2;
                              } else {
                                  return 2;
                              }
                          }
                      }
                  } else {
                      if (features[9] < -0.714099f) {
                          if (features[14] < 0.105816f) {
                              if (features[1] < 0.623286f) {
                                  if (features[4] < 2.316857f) {
                                      return 2;
                                  } else {
                                      return 3;
                                  }
                              } else {
                                  if (features[5] < -0.048503f) {
                                      if (features[10] < 0.994398f) {
                                          return 0;
                                      } else {
                                          return 0;
                                      }
                                  } else {
                                      return 2;
                                  }
                              }
                          } else {
                              if (features[4] < 2.478581f) {
                                  if (features[3] < 1.331818f) {
                                      return 2;
                                  } else {
                                      return 2;
                                  }
                              } else {
                                  return 2;
                              }
                          }
                      } else {
                          if (features[3] < 1.515861f) {
                              if (features[14] < 0.050827f) {
                                  if (features[2] < -0.985563f) {
                                      return 0;
                                  } else {
                                      return 0;
                                  }
                              } else {
                                  return 3;
                              }
                          } else {
                              if (features[0] < -0.013030f) {
                                  return 3;
                              } else {
                                  return 0;
                              }
                          }
                      }
                  }
              }
          }
        }
        

static inline int32_t random_forest_tree_5(const float *features, int32_t features_length) {
          if (features[15] < -0.130487f) {
              if (features[7] < 0.344787f) {
                  if (features[5] < -0.008344f) {
                      if (features[0] < -0.085591f) {
                          if (features[2] < -0.802574f) {
                              return 2;
                          } else {
                              return 2;
                          }
                      } else {
                          if (features[10] < 0.086475f) {
                              if (features[1] < 0.241704f) {
                                  if (features[1] < 0.178836f) {
                                      return 3;
                                  } else {
                                      if (features[8] < 0.235321f) {
                                          return 0;
                                      } else {
                                          return 0;
                                      }
                                  }
                              } else {
                                  return 1;
                              }
                          } else {
                              if (features[6] < 0.170371f) {
                                  if (features[15] < -0.205179f) {
                                      if (features[11] < 0.516204f) {
                                          if (features[6] < 0.068907f) {
                                              if (features[8] < 0.124567f) {
                                                  return 1;
                                              } else {
                                                  if (features[12] < -0.297438f) {
                                                      if (features[11] < 0.511574f) {
                                                          return 0;
                                                      } else {
                                                          return 1;
                                                      }
                                                  } else {
                                                      return 0;
                                                  }
                                              }
                                          } else {
                                              if (features[3] < 0.847854f) {
                                                  if (features[8] < 0.154792f) {
                                                      return 2;
                                                  } else {
                                                      return 2;
                                                  }
                                              } else {
                                                  if (features[2] < -0.621095f) {
                                                      if (features[8] < 0.237837f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      return 1;
                                                  }
                                              }
                                          }
                                      } else {
                                          if (features[11] < 0.780093f) {
                                              if (features[4] < 1.386372f) {
                                                  if (features[12] < -0.522295f) {
                                                      return 1;
                                                  } else {
                                                      if (features[6] < 0.042901f) {
                                                          return 1;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              } else {
                                                  if (features[15] < -0.228577f) {
                                                      if (features[13] < 0.184170f) {
                                                          return 1;
                                                      } else {
                                                          return 1;
                                                      }
                                                  } else {
                                                      return 1;
                                                  }
                                              }
                                          } else {
                                              return 1;
                                          }
                                      }
                                  } else {
                                      if (features[14] < 0.209472f) {
                                          if (features[1] < 0.384575f) {
                                              if (features[13] < -0.339444f) {
                                                  return 0;
                                              } else {
                                                  if (features[9] < -0.390738f) {
                                                      return 0;
                                                  } else {
                                                      return 0;
                                                  }
                                              }
                                          } else {
                                              if (features[4] < 2.817956f) {
                                                  return 3;
                                              } else {
                                                  return 1;
                                              }
                                          }
                                      } else {
                                          if (features[11] < 0.516204f) {
                                              return 1;
                                          } else {
                                              return 1;
                                          }
                                      }
                                  }
                              } else {
                                  if (features[3] < 1.846752f) {
                                      if (features[8] < 0.426562f) {
                                          if (features[14] < 0.128977f) {
                                              return 0;
                                          } else {
                                              if (features[9] < -0.665691f) {
                                                  return 0;
                                              } else {
                                                  return 2;
                                              }
                                          }
                                      } else {
                                          return 1;
                                      }
                                  } else {
                                      return 3;
                                  }
                              }
                          }
                      }
                  } else {
                      if (features[3] < 1.021121f) {
                          if (features[12] < -0.729991f) {
                              if (features[1] < 0.227567f) {
                                  return 1;
                              } else {
                                  return 2;
                              }
                          } else {
                              if (features[15] < -0.735977f) {
                                  if (features[12] < -0.075880f) {
                                      return 1;
                                  } else {
                                      return 1;
                                  }
                              } else {
                                  if (features[11] < 0.817130f) {
                                      if (features[3] < 0.518943f) {
                                          if (features[9] < -0.178946f) {
                                              if (features[1] < 0.234474f) {
                                                  return 1;
                                              } else {
                                                  return 2;
                                              }
                                          } else {
                                              if (features[9] < -0.156176f) {
                                                  return 3;
                                              } else {
                                                  if (features[10] < 0.170907f) {
                                                      return 0;
                                                  } else {
                                                      return 0;
                                                  }
                                              }
                                          }
                                      } else {
                                          if (features[8] < 0.115338f) {
                                              if (features[7] < 0.270515f) {
                                                  if (features[14] < -0.029439f) {
                                                      return 2;
                                                  } else {
                                                      if (features[13] < -0.037752f) {
                                                          return 2;
                                                      } else {
                                                          return 2;
                                                      }
                                                  }
                                              } else {
                                                  return 2;
                                              }
                                          } else {
                                              if (features[14] < 0.079900f) {
                                                  if (features[15] < -0.414901f) {
                                                      return 2;
                                                  } else {
                                                      return 0;
                                                  }
                                              } else {
                                                  if (features[12] < -0.324356f) {
                                                      if (features[4] < 1.626736f) {
                                                          return 2;
                                                      } else {
                                                          return 2;
                                                      }
                                                  } else {
                                                      return 2;
                                                  }
                                              }
                                          }
                                      }
                                  } else {
                                      if (features[9] < -0.163385f) {
                                          if (features[14] < 0.076625f) {
                                              if (features[13] < -0.455614f) {
                                                  if (features[14] < -0.122951f) {
                                                      return 2;
                                                  } else {
                                                      return 2;
                                                  }
                                              } else {
                                                  return 2;
                                              }
                                          } else {
                                              if (features[14] < 0.362254f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          }
                                      } else {
                                          if (features[6] < 0.045853f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      }
                                  }
                              }
                          }
                      } else {
                          if (features[12] < 0.003723f) {
                              if (features[9] < -0.440062f) {
                                  if (features[8] < 0.406966f) {
                                      if (features[1] < 0.416130f) {
                                          return 0;
                                      } else {
                                          if (features[14] < 0.509063f) {
                                              return 0;
                                          } else {
                                              return 2;
                                          }
                                      }
                                  } else {
                                      return 1;
                                  }
                              } else {
                                  if (features[9] < -0.110045f) {
                                      if (features[15] < -0.243578f) {
                                          if (features[2] < -0.535677f) {
                                              if (features[7] < 0.209973f) {
                                                  return 1;
                                              } else {
                                                  if (features[14] < 0.016406f) {
                                                      return 1;
                                                  } else {
                                                      return 1;
                                                  }
                                              }
                                          } else {
                                              return 1;
                                          }
                                      } else {
                                          if (features[13] < -0.019600f) {
                                              return 0;
                                          } else {
                                              return 1;
                                          }
                                      }
                                  } else {
                                      return 0;
                                  }
                              }
                          } else {
                              return 3;
                          }
                      }
                  }
              } else {
                  if (features[12] < -0.078005f) {
                      if (features[13] < -0.023230f) {
                          if (features[2] < -1.008929f) {
                              if (features[5] < 0.168329f) {
                                  if (features[10] < 0.524952f) {
                                      return 2;
                                  } else {
                                      if (features[14] < 0.149781f) {
                                          if (features[0] < -0.109751f) {
                                              return 2;
                                          } else {
                                              if (features[2] < -1.920917f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          }
                                      } else {
                                          return 2;
                                      }
                                  }
                              } else {
                                  if (features[10] < 0.824272f) {
                                      return 2;
                                  } else {
                                      if (features[3] < 1.287007f) {
                                          return 2;
                                      } else {
                                          return 2;
                                      }
                                  }
                              }
                          } else {
                              if (features[12] < -0.427175f) {
                                  return 2;
                              } else {
                                  return 2;
                              }
                          }
                      } else {
                          if (features[10] < 0.367901f) {
                              if (features[7] < 0.369797f) {
                                  return 2;
                              } else {
                                  if (features[2] < -1.177843f) {
                                      if (features[2] < -1.371235f) {
                                          return 2;
                                      } else {
                                          return 2;
                                      }
                                  } else {
                                      return 2;
                                  }
                              }
                          } else {
                              if (features[2] < -0.964043f) {
                                  if (features[1] < 0.539346f) {
                                      if (features[4] < 2.546363f) {
                                          if (features[0] < -0.055855f) {
                                              if (features[7] < 0.421509f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          } else {
                                              return 2;
                                          }
                                      } else {
                                          return 2;
                                      }
                                  } else {
                                      if (features[11] < 0.354167f) {
                                          return 2;
                                      } else {
                                          if (features[1] < 0.705791f) {
                                              if (features[1] < 0.682195f) {
                                                  if (features[5] < 0.167134f) {
                                                      return 2;
                                                  } else {
                                                      return 2;
                                                  }
                                              } else {
                                                  return 2;
                                              }
                                          } else {
                                              return 2;
                                          }
                                      }
                                  }
                              } else {
                                  if (features[15] < -0.514654f) {
                                      return 2;
                                  } else {
                                      if (features[9] < -0.779158f) {
                                          if (features[13] < 0.228355f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      } else {
                                          return 2;
                                      }
                                  }
                              }
                          }
                      }
                  } else {
                      return 3;
                  }
              }
          } else {
              if (features[10] < 0.424498f) {
                  if (features[12] < -0.066723f) {
                      if (features[2] < -0.588949f) {
                          if (features[5] < -0.073376f) {
                              return 3;
                          } else {
                              if (features[4] < 1.145363f) {
                                  if (features[11] < 0.671296f) {
                                      return 0;
                                  } else {
                                      return 2;
                                  }
                              } else {
                                  if (features[13] < 0.053813f) {
                                      if (features[7] < 0.151320f) {
                                          if (features[12] < -0.096233f) {
                                              if (features[2] < -0.965941f) {
                                                  return 0;
                                              } else {
                                                  return 1;
                                              }
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          return 0;
                                      }
                                  } else {
                                      if (features[9] < -0.212765f) {
                                          if (features[3] < 1.184348f) {
                                              return 2;
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          return 1;
                                      }
                                  }
                              }
                          }
                      } else {
                          if (features[13] < -0.177786f) {
                              if (features[6] < 0.116728f) {
                                  if (features[3] < 0.773812f) {
                                      return 2;
                                  } else {
                                      if (features[2] < -0.259358f) {
                                          return 0;
                                      } else {
                                          return 1;
                                      }
                                  }
                              } else {
                                  if (features[10] < 0.198086f) {
                                      return 3;
                                  } else {
                                      return 3;
                                  }
                              }
                          } else {
                              if (features[5] < -0.060046f) {
                                  return 3;
                              } else {
                                  if (features[10] < 0.285127f) {
                                      if (features[8] < 0.075443f) {
                                          return 3;
                                      } else {
                                          if (features[9] < -0.228892f) {
                                              if (features[13] < 0.042748f) {
                                                  return 0;
                                              } else {
                                                  return 0;
                                              }
                                          } else {
                                              if (features[15] < -0.065182f) {
                                                  if (features[8] < 0.232637f) {
                                                      return 0;
                                                  } else {
                                                      return 1;
                                                  }
                                              } else {
                                                  return 0;
                                              }
                                          }
                                      }
                                  } else {
                                      if (features[8] < 0.159714f) {
                                          return 2;
                                      } else {
                                          return 0;
                                      }
                                  }
                              }
                          }
                      }
                  } else {
                      if (features[3] < 1.833007f) {
                          if (features[0] < -0.065141f) {
                              if (features[3] < 1.653240f) {
                                  if (features[3] < 1.600062f) {
                                      if (features[5] < -0.040191f) {
                                          return 0;
                                      } else {
                                          return 2;
                                      }
                                  } else {
                                      return 3;
                                  }
                              } else {
                                  return 0;
                              }
                          } else {
                              if (features[11] < 0.962963f) {
                                  if (features[13] < -0.165253f) {
                                      if (features[11] < 0.516204f) {
                                          if (features[2] < -0.301073f) {
                                              return 0;
                                          } else {
                                              if (features[13] < -0.234668f) {
                                                  return 0;
                                              } else {
                                                  return 3;
                                              }
                                          }
                                      } else {
                                          if (features[15] < -0.043622f) {
                                              if (features[7] < 0.153362f) {
                                                  return 0;
                                              } else {
                                                  return 3;
                                              }
                                          } else {
                                              return 0;
                                          }
                                      }
                                  } else {
                                      if (features[6] < 0.142747f) {
                                          if (features[14] < 0.142038f) {
                                              if (features[5] < -0.089813f) {
                                                  return 3;
                                              } else {
                                                  if (features[7] < 0.252756f) {
                                                      if (features[14] < -0.182126f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      return 3;
                                                  }
                                              }
                                          } else {
                                              if (features[15] < -0.052072f) {
                                                  return 1;
                                              } else {
                                                  if (features[0] < 0.024451f) {
                                                      return 0;
                                                  } else {
                                                      return 0;
                                                  }
                                              }
                                          }
                                      } else {
                                          return 0;
                                      }
                                  }
                              } else {
                                  return 2;
                              }
                          }
                      } else {
                          if (features[13] < -0.073404f) {
                              if (features[2] < -0.631510f) {
                                  return 0;
                              } else {
                                  if (features[14] < -0.351961f) {
                                      return 0;
                                  } else {
                                      if (features[2] < -0.405385f) {
                                          return 3;
                                      } else {
                                          if (features[9] < -0.275765f) {
                                              return 3;
                                          } else {
                                              return 3;
                                          }
                                      }
                                  }
                              }
                          } else {
                              if (features[8] < 0.419541f) {
                                  return 3;
                              } else {
                                  if (features[15] < -0.095392f) {
                                      if (features[4] < 2.976042f) {
                                          return 3;
                                      } else {
                                          return 0;
                                      }
                                  } else {
                                      if (features[4] < 2.744451f) {
                                          if (features[7] < 0.318574f) {
                                              if (features[7] < 0.292556f) {
                                                  return 0;
                                              } else {
                                                  if (features[2] < -0.643642f) {
                                                      return 0;
                                                  } else {
                                                      return 3;
                                                  }
                                              }
                                          } else {
                                              if (features[1] < 0.517349f) {
                                                  return 3;
                                              } else {
                                                  return 0;
                                              }
                                          }
                                      } else {
                                          return 0;
                                      }
                                  }
                              }
                          }
                      }
                  }
              } else {
                  if (features[3] < 1.328464f) {
                      if (features[8] < 0.276203f) {
                          if (features[1] < 0.431933f) {
                              if (features[12] < 0.064571f) {
                                  if (features[15] < -0.037805f) {
                                      return 0;
                                  } else {
                                      return 0;
                                  }
                              } else {
                                  return 0;
                              }
                          } else {
                              if (features[10] < 0.785095f) {
                                  if (features[6] < 0.254723f) {
                                      return 2;
                                  } else {
                                      if (features[14] < -0.126411f) {
                                          return 2;
                                      } else {
                                          return 2;
                                      }
                                  }
                              } else {
                                  return 2;
                              }
                          }
                      } else {
                          if (features[12] < -0.049768f) {
                              if (features[14] < 0.072400f) {
                                  return 3;
                              } else {
                                  return 2;
                              }
                          } else {
                              if (features[2] < -0.919363f) {
                                  return 3;
                              } else {
                                  return 3;
                              }
                          }
                      }
                  } else {
                      if (features[1] < 0.621639f) {
                          if (features[8] < 0.299212f) {
                              if (features[10] < 0.549185f) {
                                  return 0;
                              } else {
                                  if (features[3] < 1.391862f) {
                                      return 2;
                                  } else {
                                      return 2;
                                  }
                              }
                          } else {
                              if (features[4] < 2.849592f) {
                                  if (features[7] < 0.273685f) {
                                      return 0;
                                  } else {
                                      if (features[7] < 0.422303f) {
                                          if (features[1] < 0.486318f) {
                                              return 3;
                                          } else {
                                              if (features[2] < -0.901908f) {
                                                  return 3;
                                              } else {
                                                  if (features[8] < 0.374596f) {
                                                      return 3;
                                                  } else {
                                                      return 3;
                                                  }
                                              }
                                          }
                                      } else {
                                          return 3;
                                      }
                                  }
                              } else {
                                  if (features[10] < 0.460424f) {
                                      return 0;
                                  } else {
                                      return 0;
                                  }
                              }
                          }
                      } else {
                          if (features[13] < -0.071340f) {
                              return 2;
                          } else {
                              if (features[0] < -0.119031f) {
                                  return 2;
                              } else {
                                  return 0;
                              }
                          }
                      }
                  }
              }
          }
        }
        

static inline int32_t random_forest_tree_6(const float *features, int32_t features_length) {
          if (features[7] < 0.334154f) {
              if (features[15] < -0.153309f) {
                  if (features[3] < 0.946600f) {
                      if (features[11] < 0.812500f) {
                          if (features[15] < -0.675651f) {
                              if (features[10] < 0.138543f) {
                                  return 1;
                              } else {
                                  if (features[15] < -0.722616f) {
                                      if (features[6] < 0.102184f) {
                                          if (features[14] < 0.914938f) {
                                              return 1;
                                          } else {
                                              return 1;
                                          }
                                      } else {
                                          if (features[3] < 0.702253f) {
                                              return 1;
                                          } else {
                                              return 1;
                                          }
                                      }
                                  } else {
                                      return 1;
                                  }
                              }
                          } else {
                              if (features[8] < 0.075623f) {
                                  if (features[7] < 0.148312f) {
                                      return 1;
                                  } else {
                                      return 2;
                                  }
                              } else {
                                  if (features[7] < 0.193113f) {
                                      if (features[4] < 1.435154f) {
                                          if (features[12] < -0.160718f) {
                                              if (features[3] < 0.584418f) {
                                                  if (features[9] < -0.239608f) {
                                                      return 0;
                                                  } else {
                                                      return 1;
                                                  }
                                              } else {
                                                  if (features[3] < 0.723899f) {
                                                      return 0;
                                                  } else {
                                                      if (features[9] < -0.176937f) {
                                                          return 0;
                                                      } else {
                                                          return 1;
                                                      }
                                                  }
                                              }
                                          } else {
                                              if (features[3] < 0.325671f) {
                                                  return 0;
                                              } else {
                                                  return 0;
                                              }
                                          }
                                      } else {
                                          return 1;
                                      }
                                  } else {
                                      if (features[0] < 0.079361f) {
                                          if (features[11] < 0.516204f) {
                                              if (features[7] < 0.256363f) {
                                                  if (features[5] < -0.013222f) {
                                                      if (features[0] < -0.048211f) {
                                                          return 2;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      if (features[4] < 1.393432f) {
                                                          return 2;
                                                      } else {
                                                          return 2;
                                                      }
                                                  }
                                              } else {
                                                  if (features[8] < 0.149258f) {
                                                      return 2;
                                                  } else {
                                                      if (features[3] < 0.809403f) {
                                                          return 2;
                                                      } else {
                                                          return 2;
                                                      }
                                                  }
                                              }
                                          } else {
                                              if (features[0] < 0.008099f) {
                                                  return 1;
                                              } else {
                                                  if (features[13] < 0.006773f) {
                                                      return 0;
                                                  } else {
                                                      return 2;
                                                  }
                                              }
                                          }
                                      } else {
                                          if (features[6] < 0.142771f) {
                                              return 0;
                                          } else {
                                              return 2;
                                          }
                                      }
                                  }
                              }
                          }
                      } else {
                          if (features[12] < -0.138122f) {
                              return 2;
                          } else {
                              if (features[9] < -0.159476f) {
                                  return 2;
                              } else {
                                  return 3;
                              }
                          }
                      }
                  } else {
                      if (features[11] < 0.516204f) {
                          if (features[14] < 0.672070f) {
                              if (features[5] < -0.082824f) {
                                  return 2;
                              } else {
                                  if (features[1] < 0.310920f) {
                                      if (features[9] < -0.197921f) {
                                          if (features[5] < 0.001650f) {
                                              if (features[10] < 0.141853f) {
                                                  return 0;
                                              } else {
                                                  return 0;
                                              }
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          return 1;
                                      }
                                  } else {
                                      if (features[15] < -0.511498f) {
                                          return 2;
                                      } else {
                                          if (features[2] < -0.960850f) {
                                              return 0;
                                          } else {
                                              if (features[5] < 0.063499f) {
                                                  return 0;
                                              } else {
                                                  return 0;
                                              }
                                          }
                                      }
                                  }
                              }
                          } else {
                              if (features[7] < 0.272269f) {
                                  return 1;
                              } else {
                                  return 2;
                              }
                          }
                      } else {
                          if (features[15] < -0.262797f) {
                              if (features[7] < 0.115743f) {
                                  return 1;
                              } else {
                                  if (features[8] < 0.350381f) {
                                      return 1;
                                  } else {
                                      if (features[5] < 0.006312f) {
                                          if (features[6] < 0.141103f) {
                                              if (features[8] < 0.430547f) {
                                                  return 1;
                                              } else {
                                                  if (features[0] < 0.004866f) {
                                                      return 1;
                                                  } else {
                                                      return 1;
                                                  }
                                              }
                                          } else {
                                              return 1;
                                          }
                                      } else {
                                          if (features[2] < -1.141365f) {
                                              return 1;
                                          } else {
                                              return 1;
                                          }
                                      }
                                  }
                              }
                          } else {
                              if (features[4] < 2.159136f) {
                                  if (features[10] < 0.137953f) {
                                      return 0;
                                  } else {
                                      if (features[3] < 1.265651f) {
                                          if (features[12] < -0.160770f) {
                                              return 1;
                                          } else {
                                              return 1;
                                          }
                                      } else {
                                          return 1;
                                      }
                                  }
                              } else {
                                  if (features[0] < -0.015391f) {
                                      return 3;
                                  } else {
                                      return 0;
                                  }
                              }
                          }
                      }
                  }
              } else {
                  if (features[15] < -0.032523f) {
                      if (features[5] < -0.065568f) {
                          if (features[3] < 1.640455f) {
                              if (features[7] < 0.166447f) {
                                  return 1;
                              } else {
                                  if (features[10] < 0.340911f) {
                                      return 0;
                                  } else {
                                      return 2;
                                  }
                              }
                          } else {
                              if (features[8] < 0.531630f) {
                                  if (features[9] < -0.404538f) {
                                      if (features[4] < 2.563959f) {
                                          return 3;
                                      } else {
                                          return 3;
                                      }
                                  } else {
                                      return 3;
                                  }
                              } else {
                                  return 0;
                              }
                          }
                      } else {
                          if (features[13] < -0.154558f) {
                              if (features[1] < 0.306873f) {
                                  if (features[9] < -0.257254f) {
                                      return 2;
                                  } else {
                                      if (features[3] < 1.386571f) {
                                          if (features[8] < 0.184393f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          return 1;
                                      }
                                  }
                              } else {
                                  if (features[3] < 1.715544f) {
                                      return 0;
                                  } else {
                                      if (features[8] < 0.492764f) {
                                          if (features[2] < -0.608696f) {
                                              return 0;
                                          } else {
                                              return 3;
                                          }
                                      } else {
                                          return 0;
                                      }
                                  }
                              }
                          } else {
                              if (features[4] < 1.086591f) {
                                  if (features[9] < -0.179802f) {
                                      if (features[13] < 0.016705f) {
                                          return 2;
                                      } else {
                                          if (features[9] < -0.229765f) {
                                              if (features[0] < 0.055976f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          } else {
                                              return 2;
                                          }
                                      }
                                  } else {
                                      if (features[0] < 0.035235f) {
                                          return 0;
                                      } else {
                                          return 2;
                                      }
                                  }
                              } else {
                                  if (features[14] < 0.141745f) {
                                      if (features[3] < 1.868638f) {
                                          if (features[3] < 0.654777f) {
                                              if (features[7] < 0.204128f) {
                                                  return 0;
                                              } else {
                                                  if (features[7] < 0.226121f) {
                                                      return 2;
                                                  } else {
                                                      return 0;
                                                  }
                                              }
                                          } else {
                                              if (features[12] < 0.142015f) {
                                                  if (features[8] < 0.145540f) {
                                                      return 0;
                                                  } else {
                                                      return 0;
                                                  }
                                              } else {
                                                  return 0;
                                              }
                                          }
                                      } else {
                                          if (features[3] < 1.880358f) {
                                              return 3;
                                          } else {
                                              if (features[9] < -0.502159f) {
                                                  if (features[1] < 0.500218f) {
                                                      if (features[10] < 0.174180f) {
                                                          return 0;
                                                      } else {
                                                          return 3;
                                                      }
                                                  } else {
                                                      return 0;
                                                  }
                                              } else {
                                                  if (features[14] < -0.024935f) {
                                                      if (features[0] < 0.010931f) {
                                                          return 0;
                                                      } else {
                                                          return 3;
                                                      }
                                                  } else {
                                                      return 0;
                                                  }
                                              }
                                          }
                                      }
                                  } else {
                                      if (features[10] < 0.272770f) {
                                          if (features[4] < 1.425662f) {
                                              return 0;
                                          } else {
                                              if (features[12] < -0.061554f) {
                                                  if (features[8] < 0.363057f) {
                                                      return 0;
                                                  } else {
                                                      if (features[2] < -0.734997f) {
                                                          return 1;
                                                      } else {
                                                          return 1;
                                                      }
                                                  }
                                              } else {
                                                  if (features[13] < 0.221979f) {
                                                      return 0;
                                                  } else {
                                                      return 1;
                                                  }
                                              }
                                          }
                                      } else {
                                          if (features[13] < 0.169759f) {
                                              return 2;
                                          } else {
                                              return 0;
                                          }
                                      }
                                  }
                              }
                          }
                      }
                  } else {
                      if (features[14] < -0.187168f) {
                          if (features[3] < 1.902178f) {
                              if (features[5] < -0.030544f) {
                                  if (features[8] < 0.391744f) {
                                      return 0;
                                  } else {
                                      if (features[8] < 0.460559f) {
                                          return 3;
                                      } else {
                                          return 0;
                                      }
                                  }
                              } else {
                                  if (features[11] < 0.520833f) {
                                      if (features[4] < 0.854068f) {
                                          return 0;
                                      } else {
                                          if (features[11] < 0.511574f) {
                                              if (features[8] < 0.137616f) {
                                                  return 0;
                                              } else {
                                                  return 0;
                                              }
                                          } else {
                                              return 0;
                                          }
                                      }
                                  } else {
                                      if (features[3] < 0.783346f) {
                                          return 3;
                                      } else {
                                          return 0;
                                      }
                                  }
                              }
                          } else {
                              if (features[15] < 0.126510f) {
                                  if (features[2] < -0.637709f) {
                                      if (features[9] < -0.496768f) {
                                          return 3;
                                      } else {
                                          return 0;
                                      }
                                  } else {
                                      if (features[13] < -0.472166f) {
                                          return 0;
                                      } else {
                                          if (features[15] < 0.010083f) {
                                              return 3;
                                          } else {
                                              return 3;
                                          }
                                      }
                                  }
                              } else {
                                  return 0;
                              }
                          }
                      } else {
                          if (features[11] < 0.914352f) {
                              if (features[14] < 0.208146f) {
                                  if (features[0] < 0.003722f) {
                                      if (features[5] < -0.100285f) {
                                          return 3;
                                      } else {
                                          if (features[8] < 0.416313f) {
                                              if (features[2] < -0.309505f) {
                                                  return 0;
                                              } else {
                                                  if (features[2] < -0.307540f) {
                                                      return 0;
                                                  } else {
                                                      if (features[3] < 0.791630f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              }
                                          } else {
                                              if (features[2] < -0.529482f) {
                                                  if (features[9] < -0.575660f) {
                                                      if (features[5] < -0.028371f) {
                                                          return 3;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      if (features[15] < -0.028849f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              } else {
                                                  if (features[9] < -0.398170f) {
                                                      return 0;
                                                  } else {
                                                      return 3;
                                                  }
                                              }
                                          }
                                      }
                                  } else {
                                      if (features[11] < 0.689815f) {
                                          return 0;
                                      } else {
                                          return 0;
                                      }
                                  }
                              } else {
                                  if (features[15] < 0.025163f) {
                                      if (features[11] < 0.516204f) {
                                          if (features[5] < -0.011961f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          return 1;
                                      }
                                  } else {
                                      return 0;
                                  }
                              }
                          } else {
                              return 3;
                          }
                      }
                  }
              }
          } else {
              if (features[0] < -0.073088f) {
                  if (features[11] < 0.539352f) {
                      if (features[12] < -0.035065f) {
                          if (features[9] < -0.884561f) {
                              if (features[12] < -0.054201f) {
                                  if (features[11] < 0.354167f) {
                                      if (features[2] < -1.966505f) {
                                          return 2;
                                      } else {
                                          return 2;
                                      }
                                  } else {
                                      if (features[14] < 0.089545f) {
                                          if (features[6] < 0.356707f) {
                                              if (features[2] < -1.053819f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          } else {
                                              return 2;
                                          }
                                      } else {
                                          if (features[10] < 0.389793f) {
                                              return 2;
                                          } else {
                                              if (features[3] < 1.126672f) {
                                                  if (features[3] < 1.102921f) {
                                                      return 2;
                                                  } else {
                                                      return 2;
                                                  }
                                              } else {
                                                  return 2;
                                              }
                                          }
                                      }
                                  }
                              } else {
                                  return 0;
                              }
                          } else {
                              if (features[14] < 0.097959f) {
                                  if (features[8] < 0.268133f) {
                                      if (features[15] < -0.147842f) {
                                          return 2;
                                      } else {
                                          return 2;
                                      }
                                  } else {
                                      return 3;
                                  }
                              } else {
                                  if (features[5] < -0.032828f) {
                                      if (features[11] < 0.479167f) {
                                          return 2;
                                      } else {
                                          if (features[3] < 1.092362f) {
                                              return 2;
                                          } else {
                                              if (features[13] < 0.060369f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          }
                                      }
                                  } else {
                                      return 2;
                                  }
                              }
                          }
                      } else {
                          if (features[2] < -1.340313f) {
                              if (features[9] < -1.102464f) {
                                  return 0;
                              } else {
                                  return 0;
                              }
                          } else {
                              if (features[7] < 0.376950f) {
                                  return 0;
                              } else {
                                  return 3;
                              }
                          }
                      }
                  } else {
                      return 3;
                  }
              } else {
                  if (features[12] < -0.169433f) {
                      if (features[15] < -0.715671f) {
                          return 2;
                      } else {
                          if (features[2] < -1.159242f) {
                              if (features[5] < 0.229203f) {
                                  if (features[11] < 0.502315f) {
                                      return 2;
                                  } else {
                                      if (features[15] < -0.295646f) {
                                          return 2;
                                      } else {
                                          if (features[14] < 0.412992f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      }
                                  }
                              } else {
                                  if (features[7] < 0.496419f) {
                                      return 2;
                                  } else {
                                      if (features[10] < 1.054861f) {
                                          return 2;
                                      } else {
                                          return 2;
                                      }
                                  }
                              }
                          } else {
                              if (features[3] < 1.454765f) {
                                  if (features[10] < 0.499090f) {
                                      return 2;
                                  } else {
                                      if (features[4] < 2.245136f) {
                                          if (features[9] < -0.679325f) {
                                              return 2;
                                          } else {
                                              if (features[12] < -0.457116f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          }
                                      } else {
                                          return 2;
                                      }
                                  }
                              } else {
                                  return 2;
                              }
                          }
                      }
                  } else {
                      if (features[3] < 1.976780f) {
                          if (features[8] < 0.303076f) {
                              if (features[3] < 1.085393f) {
                                  return 1;
                              } else {
                                  if (features[12] < 0.028356f) {
                                      if (features[0] < -0.033548f) {
                                          if (features[3] < 1.274165f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      } else {
                                          return 0;
                                      }
                                  } else {
                                      return 0;
                                  }
                              }
                          } else {
                              if (features[10] < 0.902862f) {
                                  if (features[5] < 0.011179f) {
                                      if (features[4] < 2.604522f) {
                                          if (features[13] < -0.089309f) {
                                              if (features[7] < 0.409730f) {
                                                  return 3;
                                              } else {
                                                  return 3;
                                              }
                                          } else {
                                              return 3;
                                          }
                                      } else {
                                          return 3;
                                      }
                                  } else {
                                      return 3;
                                  }
                              } else {
                                  return 2;
                              }
                          }
                      } else {
                          if (features[13] < -0.049486f) {
                              return 1;
                          } else {
                              return 0;
                          }
                      }
                  }
              }
          }
        }
        

static inline int32_t random_forest_tree_7(const float *features, int32_t features_length) {
          if (features[10] < 0.414713f) {
              if (features[3] < 1.801071f) {
                  if (features[8] < 0.404910f) {
                      if (features[15] < -0.107758f) {
                          if (features[6] < 0.074277f) {
                              if (features[15] < -0.641618f) {
                                  if (features[12] < -0.417547f) {
                                      if (features[8] < 0.079743f) {
                                          return 1;
                                      } else {
                                          if (features[8] < 0.143669f) {
                                              return 1;
                                          } else {
                                              return 1;
                                          }
                                      }
                                  } else {
                                      return 0;
                                  }
                              } else {
                                  if (features[8] < 0.264825f) {
                                      if (features[10] < 0.214995f) {
                                          if (features[12] < -0.318327f) {
                                              if (features[14] < 0.626325f) {
                                                  if (features[9] < -0.227255f) {
                                                      if (features[14] < 0.428952f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      return 0;
                                                  }
                                              } else {
                                                  return 1;
                                              }
                                          } else {
                                              if (features[12] < -0.139291f) {
                                                  if (features[13] < 0.232195f) {
                                                      return 0;
                                                  } else {
                                                      return 1;
                                                  }
                                              } else {
                                                  return 0;
                                              }
                                          }
                                      } else {
                                          if (features[5] < -0.004041f) {
                                              if (features[9] < -0.227403f) {
                                                  if (features[8] < 0.154297f) {
                                                      return 2;
                                                  } else {
                                                      return 0;
                                                  }
                                              } else {
                                                  return 1;
                                              }
                                          } else {
                                              if (features[4] < 1.088846f) {
                                                  if (features[11] < 0.814815f) {
                                                      return 2;
                                                  } else {
                                                      if (features[1] < 0.176802f) {
                                                          return 2;
                                                      } else {
                                                          return 2;
                                                      }
                                                  }
                                              } else {
                                                  if (features[15] < -0.326193f) {
                                                      if (features[13] < 0.076740f) {
                                                          return 2;
                                                      } else {
                                                          return 2;
                                                      }
                                                  } else {
                                                      return 0;
                                                  }
                                              }
                                          }
                                      }
                                  } else {
                                      if (features[11] < 0.516204f) {
                                          return 0;
                                      } else {
                                          if (features[6] < 0.049841f) {
                                              return 1;
                                          } else {
                                              return 1;
                                          }
                                      }
                                  }
                              }
                          } else {
                              if (features[3] < 0.893022f) {
                                  if (features[12] < -0.381274f) {
                                      if (features[13] < 0.062036f) {
                                          return 2;
                                      } else {
                                          if (features[4] < 1.486670f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      }
                                  } else {
                                      if (features[8] < 0.109727f) {
                                          if (features[9] < -0.451380f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      } else {
                                          if (features[7] < 0.234700f) {
                                              if (features[4] < 1.505944f) {
                                                  return 0;
                                              } else {
                                                  return 0;
                                              }
                                          } else {
                                              return 2;
                                          }
                                      }
                                  }
                              } else {
                                  if (features[4] < 2.381513f) {
                                      if (features[15] < -0.514612f) {
                                          if (features[2] < -0.708341f) {
                                              return 2;
                                          } else {
                                              return 1;
                                          }
                                      } else {
                                          if (features[3] < 1.575271f) {
                                              if (features[8] < 0.337643f) {
                                                  if (features[0] < -0.070591f) {
                                                      return 0;
                                                  } else {
                                                      if (features[9] < -0.343299f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              } else {
                                                  if (features[12] < -0.098996f) {
                                                      return 1;
                                                  } else {
                                                      return 0;
                                                  }
                                              }
                                          } else {
                                              return 3;
                                          }
                                      }
                                  } else {
                                      if (features[13] < 0.098291f) {
                                          return 2;
                                      } else {
                                          if (features[8] < 0.301793f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      }
                                  }
                              }
                          }
                      } else {
                          if (features[4] < 1.078988f) {
                              if (features[9] < -0.222533f) {
                                  if (features[10] < 0.232655f) {
                                      if (features[7] < 0.120890f) {
                                          return 0;
                                      } else {
                                          return 0;
                                      }
                                  } else {
                                      if (features[15] < 0.058535f) {
                                          if (features[0] < 0.047635f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      } else {
                                          return 3;
                                      }
                                  }
                              } else {
                                  if (features[11] < 0.891204f) {
                                      if (features[12] < -0.117825f) {
                                          return 0;
                                      } else {
                                          if (features[7] < 0.068863f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      }
                                  } else {
                                      return 3;
                                  }
                              }
                          } else {
                              if (features[11] < 0.645833f) {
                                  if (features[5] < -0.093502f) {
                                      if (features[9] < -0.433154f) {
                                          return 0;
                                      } else {
                                          return 3;
                                      }
                                  } else {
                                      if (features[15] < -0.044098f) {
                                          if (features[15] < -0.044453f) {
                                              if (features[8] < 0.134982f) {
                                                  return 0;
                                              } else {
                                                  if (features[4] < 2.673689f) {
                                                      if (features[2] < -0.846415f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      return 0;
                                                  }
                                              }
                                          } else {
                                              return 3;
                                          }
                                      } else {
                                          if (features[8] < 0.387479f) {
                                              if (features[2] < -0.309352f) {
                                                  return 0;
                                              } else {
                                                  if (features[4] < 1.852459f) {
                                                      if (features[2] < -0.308455f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      return 3;
                                                  }
                                              }
                                          } else {
                                              if (features[14] < 0.058052f) {
                                                  return 0;
                                              } else {
                                                  return 1;
                                              }
                                          }
                                      }
                                  }
                              } else {
                                  return 2;
                              }
                          }
                      }
                  } else {
                      if (features[15] < -0.144466f) {
                          if (features[2] < -1.108878f) {
                              if (features[7] < 0.307409f) {
                                  if (features[5] < 0.004534f) {
                                      if (features[2] < -1.388879f) {
                                          return 1;
                                      } else {
                                          return 1;
                                      }
                                  } else {
                                      return 0;
                                  }
                              } else {
                                  return 2;
                              }
                          } else {
                              if (features[3] < 1.521789f) {
                                  if (features[6] < 0.101548f) {
                                      if (features[14] < -0.005876f) {
                                          if (features[13] < -0.217601f) {
                                              return 1;
                                          } else {
                                              return 1;
                                          }
                                      } else {
                                          if (features[15] < -0.224456f) {
                                              if (features[12] < -0.315574f) {
                                                  return 1;
                                              } else {
                                                  return 1;
                                              }
                                          } else {
                                              return 1;
                                          }
                                      }
                                  } else {
                                      return 1;
                                  }
                              } else {
                                  if (features[2] < -0.528961f) {
                                      return 1;
                                  } else {
                                      return 0;
                                  }
                              }
                          }
                      } else {
                          if (features[4] < 1.877173f) {
                              if (features[5] < -0.021522f) {
                                  return 1;
                              } else {
                                  if (features[4] < 1.870535f) {
                                      return 0;
                                  } else {
                                      return 0;
                                  }
                              }
                          } else {
                              if (features[14] < 0.141745f) {
                                  if (features[15] < -0.129396f) {
                                      if (features[12] < -0.019887f) {
                                          return 0;
                                      } else {
                                          return 3;
                                      }
                                  } else {
                                      if (features[5] < -0.043488f) {
                                          if (features[11] < 0.520833f) {
                                              return 0;
                                          } else {
                                              return 3;
                                          }
                                      } else {
                                          if (features[5] < 0.008256f) {
                                              if (features[13] < -0.231707f) {
                                                  if (features[1] < 0.266374f) {
                                                      return 0;
                                                  } else {
                                                      return 0;
                                                  }
                                              } else {
                                                  if (features[10] < 0.065048f) {
                                                      return 0;
                                                  } else {
                                                      return 0;
                                                  }
                                              }
                                          } else {
                                              if (features[6] < 0.080491f) {
                                                  if (features[14] < -0.178632f) {
                                                      if (features[15] < 0.208883f) {
                                                          return 1;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      return 0;
                                                  }
                                              } else {
                                                  return 0;
                                              }
                                          }
                                      }
                                  }
                              } else {
                                  return 1;
                              }
                          }
                      }
                  }
              } else {
                  if (features[13] < -0.140410f) {
                      if (features[8] < 0.518611f) {
                          if (features[2] < -0.605090f) {
                              if (features[1] < 0.414253f) {
                                  return 0;
                              } else {
                                  return 3;
                              }
                          } else {
                              if (features[14] < -0.435039f) {
                                  return 0;
                              } else {
                                  if (features[11] < 0.516204f) {
                                      return 3;
                                  } else {
                                      if (features[0] < 0.006265f) {
                                          if (features[10] < 0.152332f) {
                                              return 3;
                                          } else {
                                              return 3;
                                          }
                                      } else {
                                          return 3;
                                      }
                                  }
                              }
                          }
                      } else {
                          if (features[15] < -0.133222f) {
                              return 1;
                          } else {
                              return 0;
                          }
                      }
                  } else {
                      if (features[13] < 0.136339f) {
                          if (features[0] < -0.014843f) {
                              if (features[14] < -0.005278f) {
                                  if (features[10] < 0.279445f) {
                                      if (features[3] < 2.061536f) {
                                          return 0;
                                      } else {
                                          return 3;
                                      }
                                  } else {
                                      return 3;
                                  }
                              } else {
                                  if (features[5] < -0.055879f) {
                                      return 3;
                                  } else {
                                      if (features[13] < -0.047477f) {
                                          return 3;
                                      } else {
                                          if (features[9] < -0.632327f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      }
                                  }
                              }
                          } else {
                              if (features[0] < 0.050962f) {
                                  if (features[12] < -0.154245f) {
                                      return 1;
                                  } else {
                                      if (features[13] < 0.067618f) {
                                          if (features[13] < -0.029552f) {
                                              if (features[2] < -0.542019f) {
                                                  return 0;
                                              } else {
                                                  return 3;
                                              }
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          if (features[2] < -0.667450f) {
                                              if (features[2] < -1.261652f) {
                                                  if (features[9] < -0.181518f) {
                                                      if (features[1] < 0.372200f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      return 1;
                                                  }
                                              } else {
                                                  return 0;
                                              }
                                          } else {
                                              return 3;
                                          }
                                      }
                                  }
                              } else {
                                  return 1;
                              }
                          }
                      } else {
                          if (features[15] < -0.085774f) {
                              if (features[11] < 0.516204f) {
                                  return 2;
                              } else {
                                  return 1;
                              }
                          } else {
                              return 0;
                          }
                      }
                  }
              }
          } else {
              if (features[13] < 0.059929f) {
                  if (features[9] < -0.438209f) {
                      if (features[3] < 1.303697f) {
                          if (features[15] < -0.014358f) {
                              if (features[14] < -0.001821f) {
                                  if (features[8] < 0.268350f) {
                                      if (features[12] < -0.482056f) {
                                          return 2;
                                      } else {
                                          if (features[14] < -0.034365f) {
                                              if (features[6] < 0.266277f) {
                                                  return 2;
                                              } else {
                                                  if (features[14] < -0.157709f) {
                                                      return 2;
                                                  } else {
                                                      return 2;
                                                  }
                                              }
                                          } else {
                                              return 2;
                                          }
                                      }
                                  } else {
                                      if (features[5] < -0.022754f) {
                                          return 3;
                                      } else {
                                          return 2;
                                      }
                                  }
                              } else {
                                  if (features[7] < 0.328476f) {
                                      return 2;
                                  } else {
                                      if (features[5] < -0.125631f) {
                                          return 2;
                                      } else {
                                          if (features[7] < 0.378917f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      }
                                  }
                              }
                          } else {
                              return 2;
                          }
                      } else {
                          if (features[7] < 0.446740f) {
                              if (features[15] < -0.211050f) {
                                  return 1;
                              } else {
                                  if (features[4] < 2.876463f) {
                                      if (features[8] < 0.299454f) {
                                          if (features[5] < -0.050834f) {
                                              return 2;
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          if (features[2] < -0.913916f) {
                                              return 3;
                                          } else {
                                              if (features[4] < 2.339634f) {
                                                  return 3;
                                              } else {
                                                  return 3;
                                              }
                                          }
                                      }
                                  } else {
                                      if (features[10] < 0.452826f) {
                                          return 0;
                                      } else {
                                          return 0;
                                      }
                                  }
                              }
                          } else {
                              if (features[12] < -0.063136f) {
                                  if (features[7] < 0.544635f) {
                                      return 2;
                                  } else {
                                      if (features[0] < 0.035992f) {
                                          return 2;
                                      } else {
                                          return 2;
                                      }
                                  }
                              } else {
                                  if (features[4] < 3.823581f) {
                                      return 3;
                                  } else {
                                      if (features[1] < 0.885751f) {
                                          return 0;
                                      } else {
                                          return 0;
                                      }
                                  }
                              }
                          }
                      }
                  } else {
                      if (features[15] < -0.711715f) {
                          if (features[7] < 0.293866f) {
                              return 1;
                          } else {
                              return 1;
                          }
                      } else {
                          if (features[5] < 0.025590f) {
                              if (features[12] < -0.072528f) {
                                  return 2;
                              } else {
                                  if (features[2] < -1.216320f) {
                                      return 0;
                                  } else {
                                      return 0;
                                  }
                              }
                          } else {
                              if (features[14] < -0.027032f) {
                                  if (features[11] < 0.807870f) {
                                      if (features[1] < 0.419957f) {
                                          return 0;
                                      } else {
                                          return 2;
                                      }
                                  } else {
                                      return 2;
                                  }
                              } else {
                                  if (features[9] < -0.117471f) {
                                      if (features[12] < -0.224172f) {
                                          return 2;
                                      } else {
                                          if (features[4] < 1.891626f) {
                                              return 0;
                                          } else {
                                              return 1;
                                          }
                                      }
                                  } else {
                                      return 3;
                                  }
                              }
                          }
                      }
                  }
              } else {
                  if (features[14] < 0.096151f) {
                      if (features[4] < 2.061600f) {
                          return 1;
                      } else {
                          if (features[14] < 0.034606f) {
                              return 0;
                          } else {
                              return 3;
                          }
                      }
                  } else {
                      if (features[1] < 0.466777f) {
                          if (features[14] < 0.255498f) {
                              return 1;
                          } else {
                              if (features[4] < 2.307581f) {
                                  if (features[2] < -0.885864f) {
                                      if (features[9] < -0.718108f) {
                                          return 2;
                                      } else {
                                          if (features[3] < 0.724355f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      }
                                  } else {
                                      if (features[1] < 0.394045f) {
                                          if (features[11] < 0.636574f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      } else {
                                          return 2;
                                      }
                                  }
                              } else {
                                  return 2;
                              }
                          }
                      } else {
                          if (features[2] < -0.889584f) {
                              if (features[0] < -0.052886f) {
                                  if (features[4] < 2.064699f) {
                                      if (features[13] < 0.284372f) {
                                          return 2;
                                      } else {
                                          if (features[1] < 0.540295f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      }
                                  } else {
                                      if (features[13] < 0.095105f) {
                                          if (features[8] < 0.212397f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      } else {
                                          if (features[5] < -0.247145f) {
                                              if (features[7] < 0.677004f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          } else {
                                              return 2;
                                          }
                                      }
                                  }
                              } else {
                                  if (features[14] < 0.331346f) {
                                      return 2;
                                  } else {
                                      if (features[3] < 1.357024f) {
                                          return 2;
                                      } else {
                                          if (features[7] < 0.511911f) {
                                              return 2;
                                          } else {
                                              if (features[4] < 3.747049f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          }
                                      }
                                  }
                              }
                          } else {
                              return 2;
                          }
                      }
                  }
              }
          }
        }
        

static inline int32_t random_forest_tree_8(const float *features, int32_t features_length) {
          if (features[12] < -0.107725f) {
              if (features[0] < -0.070097f) {
                  if (features[2] < -1.027082f) {
                      if (features[1] < 0.492827f) {
                          return 2;
                      } else {
                          if (features[15] < -0.130229f) {
                              if (features[5] < -0.297296f) {
                                  return 2;
                              } else {
                                  if (features[3] < 0.904625f) {
                                      return 2;
                                  } else {
                                      if (features[1] < 1.127932f) {
                                          if (features[13] < -0.358471f) {
                                              return 2;
                                          } else {
                                              if (features[15] < -0.148637f) {
                                                  if (features[6] < 0.299912f) {
                                                      if (features[3] < 1.129569f) {
                                                          return 2;
                                                      } else {
                                                          return 2;
                                                      }
                                                  } else {
                                                      return 2;
                                                  }
                                              } else {
                                                  return 2;
                                              }
                                          }
                                      } else {
                                          return 2;
                                      }
                                  }
                              }
                          } else {
                              if (features[2] < -1.054534f) {
                                  return 2;
                              } else {
                                  return 2;
                              }
                          }
                      }
                  } else {
                      if (features[10] < 0.708797f) {
                          if (features[9] < -0.607446f) {
                              if (features[13] < -0.073595f) {
                                  if (features[0] < -0.099982f) {
                                      return 2;
                                  } else {
                                      return 2;
                                  }
                              } else {
                                  if (features[11] < 0.511574f) {
                                      if (features[3] < 1.029610f) {
                                          if (features[10] < 0.411334f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      } else {
                                          return 2;
                                      }
                                  } else {
                                      return 2;
                                  }
                              }
                          } else {
                              return 1;
                          }
                      } else {
                          if (features[14] < -0.118720f) {
                              return 3;
                          } else {
                              if (features[4] < 2.224732f) {
                                  return 2;
                              } else {
                                  return 2;
                              }
                          }
                      }
                  }
              } else {
                  if (features[11] < 0.548611f) {
                      if (features[9] < -0.410337f) {
                          if (features[12] < -0.155518f) {
                              if (features[10] < 0.399178f) {
                                  if (features[8] < 0.337071f) {
                                      if (features[0] < -0.054991f) {
                                          if (features[4] < 1.668283f) {
                                              return 2;
                                          } else {
                                              if (features[3] < 1.307519f) {
                                                  return 0;
                                              } else {
                                                  return 2;
                                              }
                                          }
                                      } else {
                                          if (features[2] < -0.870603f) {
                                              return 2;
                                          } else {
                                              if (features[1] < 0.283591f) {
                                                  return 2;
                                              } else {
                                                  if (features[13] < 0.322917f) {
                                                      return 0;
                                                  } else {
                                                      return 2;
                                                  }
                                              }
                                          }
                                      }
                                  } else {
                                      return 1;
                                  }
                              } else {
                                  if (features[0] < 0.123259f) {
                                      if (features[3] < 1.449616f) {
                                          if (features[2] < -0.821758f) {
                                              if (features[6] < 0.268728f) {
                                                  if (features[2] < -1.115514f) {
                                                      return 2;
                                                  } else {
                                                      return 2;
                                                  }
                                              } else {
                                                  if (features[9] < -0.738237f) {
                                                      return 2;
                                                  } else {
                                                      return 2;
                                                  }
                                              }
                                          } else {
                                              return 2;
                                          }
                                      } else {
                                          return 2;
                                      }
                                  } else {
                                      if (features[2] < -1.345467f) {
                                          return 2;
                                      } else {
                                          return 0;
                                      }
                                  }
                              }
                          } else {
                              if (features[6] < 0.186466f) {
                                  if (features[4] < 1.864023f) {
                                      return 2;
                                  } else {
                                      return 3;
                                  }
                              } else {
                                  if (features[13] < -0.047739f) {
                                      if (features[11] < 0.516204f) {
                                          return 0;
                                      } else {
                                          return 3;
                                      }
                                  } else {
                                      return 2;
                                  }
                              }
                          }
                      } else {
                          if (features[7] < 0.272300f) {
                              if (features[15] < -0.107736f) {
                                  if (features[8] < 0.358884f) {
                                      if (features[10] < 0.096351f) {
                                          if (features[11] < 0.516204f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          if (features[6] < 0.043980f) {
                                              if (features[15] < -0.420563f) {
                                                  if (features[12] < -0.419816f) {
                                                      if (features[3] < 0.505984f) {
                                                          return 1;
                                                      } else {
                                                          return 1;
                                                      }
                                                  } else {
                                                      return 1;
                                                  }
                                              } else {
                                                  if (features[14] < 0.285177f) {
                                                      return 1;
                                                  } else {
                                                      return 1;
                                                  }
                                              }
                                          } else {
                                              if (features[12] < -0.624326f) {
                                                  if (features[15] < -0.686652f) {
                                                      return 1;
                                                  } else {
                                                      return 1;
                                                  }
                                              } else {
                                                  if (features[13] < 0.168061f) {
                                                      if (features[3] < 0.684215f) {
                                                          return 2;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      if (features[11] < 0.511574f) {
                                                          return 2;
                                                      } else {
                                                          return 1;
                                                      }
                                                  }
                                              }
                                          }
                                      }
                                  } else {
                                      if (features[5] < 0.010898f) {
                                          if (features[9] < -0.359845f) {
                                              return 1;
                                          } else {
                                              if (features[3] < 2.002741f) {
                                                  if (features[12] < -0.159411f) {
                                                      if (features[15] < -0.221454f) {
                                                          return 1;
                                                      } else {
                                                          return 1;
                                                      }
                                                  } else {
                                                      if (features[6] < 0.089895f) {
                                                          return 1;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              } else {
                                                  return 1;
                                              }
                                          }
                                      } else {
                                          return 1;
                                      }
                                  }
                              } else {
                                  if (features[13] < -0.298967f) {
                                      return 3;
                                  } else {
                                      if (features[1] < 0.249217f) {
                                          if (features[2] < -0.660831f) {
                                              return 1;
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          if (features[9] < -0.352907f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      }
                                  }
                              }
                          } else {
                              if (features[10] < 0.538257f) {
                                  return 1;
                              } else {
                                  return 2;
                              }
                          }
                      }
                  } else {
                      if (features[5] < -0.007288f) {
                          return 2;
                      } else {
                          if (features[9] < -0.354124f) {
                              if (features[7] < 0.240529f) {
                                  return 2;
                              } else {
                                  if (features[8] < 0.433267f) {
                                      return 2;
                                  } else {
                                      return 2;
                                  }
                              }
                          } else {
                              if (features[15] < -0.627546f) {
                                  return 1;
                              } else {
                                  if (features[11] < 0.819444f) {
                                      if (features[13] < -0.401914f) {
                                          return 2;
                                      } else {
                                          return 2;
                                      }
                                  } else {
                                      if (features[6] < 0.034713f) {
                                          return 2;
                                      } else {
                                          return 2;
                                      }
                                  }
                              }
                          }
                      }
                  }
              }
          } else {
              if (features[11] < 0.525463f) {
                  if (features[15] < -0.288922f) {
                      if (features[0] < -0.051436f) {
                          if (features[8] < 0.125038f) {
                              return 2;
                          } else {
                              return 1;
                          }
                      } else {
                          if (features[3] < 0.978455f) {
                              if (features[0] < -0.006670f) {
                                  return 1;
                              } else {
                                  return 0;
                              }
                          } else {
                              if (features[8] < 0.350162f) {
                                  return 1;
                              } else {
                                  if (features[15] < -0.350296f) {
                                      if (features[8] < 0.458306f) {
                                          return 1;
                                      } else {
                                          return 1;
                                      }
                                  } else {
                                      return 1;
                                  }
                              }
                          }
                      }
                  } else {
                      if (features[9] < -0.636627f) {
                          if (features[1] < 0.665854f) {
                              if (features[12] < 0.099641f) {
                                  if (features[14] < 0.104860f) {
                                      if (features[6] < 0.244771f) {
                                          if (features[15] < 0.068223f) {
                                              return 2;
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          if (features[4] < 2.759125f) {
                                              if (features[4] < 2.317984f) {
                                                  return 3;
                                              } else {
                                                  if (features[3] < 1.471857f) {
                                                      return 3;
                                                  } else {
                                                      if (features[10] < 0.372136f) {
                                                          return 0;
                                                      } else {
                                                          return 3;
                                                      }
                                                  }
                                              }
                                          } else {
                                              return 0;
                                          }
                                      }
                                  } else {
                                      if (features[2] < -0.831658f) {
                                          if (features[7] < 0.386018f) {
                                              return 2;
                                          } else {
                                              if (features[0] < -0.086578f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          }
                                      } else {
                                          return 0;
                                      }
                                  }
                              } else {
                                  if (features[6] < 0.252814f) {
                                      return 0;
                                  } else {
                                      return 0;
                                  }
                              }
                          } else {
                              if (features[7] < 0.584494f) {
                                  if (features[15] < -0.057027f) {
                                      return 0;
                                  } else {
                                      return 0;
                                  }
                              } else {
                                  return 0;
                              }
                          }
                      } else {
                          if (features[13] < -0.154605f) {
                              if (features[5] < -0.025207f) {
                                  if (features[2] < -0.636997f) {
                                      if (features[1] < 0.472143f) {
                                          if (features[15] < -0.094293f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          if (features[3] < 2.589103f) {
                                              return 3;
                                          } else {
                                              return 0;
                                          }
                                      }
                                  } else {
                                      if (features[4] < 2.032817f) {
                                          if (features[10] < 0.134851f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          if (features[14] < -0.420392f) {
                                              return 0;
                                          } else {
                                              if (features[3] < 1.846563f) {
                                                  return 3;
                                              } else {
                                                  if (features[9] < -0.301838f) {
                                                      return 3;
                                                  } else {
                                                      return 3;
                                                  }
                                              }
                                          }
                                      }
                                  }
                              } else {
                                  if (features[12] < -0.044651f) {
                                      if (features[1] < 0.338524f) {
                                          return 0;
                                      } else {
                                          return 3;
                                      }
                                  } else {
                                      if (features[12] < 0.015066f) {
                                          if (features[4] < 2.075611f) {
                                              return 0;
                                          } else {
                                              if (features[3] < 1.914936f) {
                                                  if (features[8] < 0.450170f) {
                                                      return 1;
                                                  } else {
                                                      if (features[5] < -0.007585f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              } else {
                                                  return 3;
                                              }
                                          }
                                      } else {
                                          if (features[14] < -0.287286f) {
                                              return 0;
                                          } else {
                                              if (features[7] < 0.191878f) {
                                                  return 0;
                                              } else {
                                                  return 0;
                                              }
                                          }
                                      }
                                  }
                              }
                          } else {
                              if (features[7] < 0.292593f) {
                                  if (features[12] < -0.058786f) {
                                      if (features[14] < 0.205197f) {
                                          if (features[5] < -0.070545f) {
                                              return 3;
                                          } else {
                                              if (features[15] < -0.203085f) {
                                                  return 2;
                                              } else {
                                                  if (features[1] < 0.386295f) {
                                                      if (features[5] < 0.039466f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      if (features[3] < 1.867845f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              }
                                          }
                                      } else {
                                          return 1;
                                      }
                                  } else {
                                      if (features[8] < 0.117932f) {
                                          if (features[1] < 0.185087f) {
                                              if (features[8] < 0.056529f) {
                                                  return 0;
                                              } else {
                                                  return 0;
                                              }
                                          } else {
                                              return 2;
                                          }
                                      } else {
                                          if (features[5] < -0.100590f) {
                                              if (features[12] < 0.010498f) {
                                                  return 3;
                                              } else {
                                                  return 0;
                                              }
                                          } else {
                                              if (features[13] < 0.421753f) {
                                                  if (features[5] < 0.049933f) {
                                                      if (features[3] < 2.064513f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      if (features[1] < 0.338233f) {
                                                          return 0;
                                                      } else {
                                                          return 3;
                                                      }
                                                  }
                                              } else {
                                                  if (features[15] < 0.015869f) {
                                                      return 1;
                                                  } else {
                                                      if (features[3] < 0.838069f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              }
                                          }
                                      }
                                  }
                              } else {
                                  if (features[5] < -0.072555f) {
                                      if (features[3] < 1.698699f) {
                                          if (features[11] < 0.506944f) {
                                              return 0;
                                          } else {
                                              return 2;
                                          }
                                      } else {
                                          return 3;
                                      }
                                  } else {
                                      if (features[6] < 0.220793f) {
                                          if (features[0] < -0.048786f) {
                                              return 0;
                                          } else {
                                              return 3;
                                          }
                                      } else {
                                          if (features[4] < 2.776215f) {
                                              if (features[1] < 0.505091f) {
                                                  if (features[7] < 0.320941f) {
                                                      return 0;
                                                  } else {
                                                      return 0;
                                                  }
                                              } else {
                                                  return 0;
                                              }
                                          } else {
                                              return 0;
                                          }
                                      }
                                  }
                              }
                          }
                      }
                  }
              } else {
                  if (features[0] < 0.025519f) {
                      if (features[8] < 0.309805f) {
                          if (features[9] < -0.478161f) {
                              return 2;
                          } else {
                              if (features[5] < 0.001945f) {
                                  return 3;
                              } else {
                                  return 0;
                              }
                          }
                      } else {
                          if (features[13] < -0.054835f) {
                              return 3;
                          } else {
                              return 3;
                          }
                      }
                  } else {
                      if (features[10] < 0.463824f) {
                          if (features[4] < 1.145404f) {
                              if (features[5] < 0.039536f) {
                                  if (features[2] < -0.698849f) {
                                      return 0;
                                  } else {
                                      return 2;
                                  }
                              } else {
                                  if (features[9] < -0.205457f) {
                                      return 0;
                                  } else {
                                      return 0;
                                  }
                              }
                          } else {
                              if (features[11] < 0.826389f) {
                                  return 0;
                              } else {
                                  return 0;
                              }
                          }
                      } else {
                          if (features[11] < 0.715278f) {
                              if (features[12] < -0.030314f) {
                                  return 0;
                              } else {
                                  return 0;
                              }
                          } else {
                              if (features[11] < 0.812500f) {
                                  if (features[0] < 0.083303f) {
                                      if (features[3] < 0.592080f) {
                                          return 1;
                                      } else {
                                          return 1;
                                      }
                                  } else {
                                      return 1;
                                  }
                              } else {
                                  return 3;
                              }
                          }
                      }
                  }
              }
          }
        }
        

static inline int32_t random_forest_tree_9(const float *features, int32_t features_length) {
          if (features[12] < -0.106220f) {
              if (features[9] < -0.387701f) {
                  if (features[15] < -0.137949f) {
                      if (features[10] < 0.329852f) {
                          if (features[8] < 0.337881f) {
                              if (features[14] < 0.576918f) {
                                  if (features[4] < 1.397213f) {
                                      return 2;
                                  } else {
                                      if (features[6] < 0.209553f) {
                                          if (features[5] < -0.052901f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          return 2;
                                      }
                                  }
                              } else {
                                  if (features[15] < -0.415638f) {
                                      if (features[14] < 0.709674f) {
                                          return 2;
                                      } else {
                                          if (features[8] < 0.181420f) {
                                              return 2;
                                          } else {
                                              return 2;
                                          }
                                      }
                                  } else {
                                      return 2;
                                  }
                              }
                          } else {
                              if (features[1] < 0.513874f) {
                                  return 1;
                              } else {
                                  return 2;
                              }
                          }
                      } else {
                          if (features[1] < 0.516343f) {
                              if (features[13] < -0.391130f) {
                                  if (features[6] < 0.231244f) {
                                      if (features[0] < -0.012087f) {
                                          return 2;
                                      } else {
                                          return 2;
                                      }
                                  } else {
                                      return 2;
                                  }
                              } else {
                                  if (features[3] < 1.277011f) {
                                      if (features[1] < 0.349320f) {
                                          return 2;
                                      } else {
                                          if (features[13] < 0.080498f) {
                                              if (features[6] < 0.235149f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          } else {
                                              if (features[2] < -0.837665f) {
                                                  if (features[8] < 0.221442f) {
                                                      if (features[8] < 0.207610f) {
                                                          return 2;
                                                      } else {
                                                          return 2;
                                                      }
                                                  } else {
                                                      return 2;
                                                  }
                                              } else {
                                                  return 2;
                                              }
                                          }
                                      }
                                  } else {
                                      if (features[9] < -0.684133f) {
                                          return 2;
                                      } else {
                                          return 0;
                                      }
                                  }
                              }
                          } else {
                              if (features[7] < 0.389520f) {
                                  return 2;
                              } else {
                                  if (features[15] < -0.624820f) {
                                      if (features[15] < -0.665196f) {
                                          return 2;
                                      } else {
                                          return 2;
                                      }
                                  } else {
                                      if (features[9] < -0.814779f) {
                                          if (features[14] < 0.236638f) {
                                              if (features[15] < -0.497758f) {
                                                  return 2;
                                              } else {
                                                  if (features[6] < 0.379418f) {
                                                      return 2;
                                                  } else {
                                                      return 2;
                                                  }
                                              }
                                          } else {
                                              if (features[4] < 2.111694f) {
                                                  if (features[6] < 0.334878f) {
                                                      return 2;
                                                  } else {
                                                      return 2;
                                                  }
                                              } else {
                                                  if (features[13] < 0.252449f) {
                                                      if (features[5] < -0.205179f) {
                                                          return 2;
                                                      } else {
                                                          return 2;
                                                      }
                                                  } else {
                                                      return 2;
                                                  }
                                              }
                                          }
                                      } else {
                                          if (features[8] < 0.264829f) {
                                              if (features[15] < -0.534987f) {
                                                  return 2;
                                              } else {
                                                  if (features[5] < 0.229724f) {
                                                      return 2;
                                                  } else {
                                                      return 2;
                                                  }
                                              }
                                          } else {
                                              if (features[6] < 0.401947f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          }
                                      }
                                  }
                              }
                          }
                      }
                  } else {
                      if (features[13] < -0.028930f) {
                          if (features[8] < 0.268400f) {
                              if (features[7] < 0.349860f) {
                                  if (features[1] < 0.312250f) {
                                      return 2;
                                  } else {
                                      return 0;
                                  }
                              } else {
                                  if (features[4] < 2.268941f) {
                                      return 2;
                                  } else {
                                      return 2;
                                  }
                              }
                          } else {
                              if (features[8] < 0.330715f) {
                                  return 3;
                              } else {
                                  if (features[4] < 2.644496f) {
                                      if (features[1] < 0.521265f) {
                                          return 3;
                                      } else {
                                          return 3;
                                      }
                                  } else {
                                      return 3;
                                  }
                              }
                          }
                      } else {
                          if (features[0] < -0.055279f) {
                              if (features[5] < -0.069450f) {
                                  return 2;
                              } else {
                                  return 2;
                              }
                          } else {
                              if (features[2] < -0.713866f) {
                                  return 2;
                              } else {
                                  return 1;
                              }
                          }
                      }
                  }
              } else {
                  if (features[5] < 0.003966f) {
                      if (features[15] < -0.144788f) {
                          if (features[0] < 0.038179f) {
                              if (features[7] < 0.109363f) {
                                  if (features[12] < -0.258257f) {
                                      return 1;
                                  } else {
                                      if (features[5] < -0.028771f) {
                                          return 3;
                                      } else {
                                          return 0;
                                      }
                                  }
                              } else {
                                  if (features[4] < 1.724173f) {
                                      if (features[15] < -0.611218f) {
                                          if (features[7] < 0.138355f) {
                                              if (features[1] < 0.189100f) {
                                                  return 1;
                                              } else {
                                                  return 1;
                                              }
                                          } else {
                                              if (features[5] < -0.043805f) {
                                                  return 1;
                                              } else {
                                                  return 1;
                                              }
                                          }
                                      } else {
                                          if (features[13] < -0.064372f) {
                                              if (features[0] < -0.004560f) {
                                                  return 0;
                                              } else {
                                                  return 1;
                                              }
                                          } else {
                                              if (features[14] < 0.638315f) {
                                                  if (features[12] < -0.294448f) {
                                                      return 0;
                                                  } else {
                                                      if (features[2] < -0.452208f) {
                                                          return 1;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              } else {
                                                  return 1;
                                              }
                                          }
                                      }
                                  } else {
                                      if (features[11] < 0.516204f) {
                                          return 1;
                                      } else {
                                          if (features[10] < 0.264145f) {
                                              if (features[2] < -0.722395f) {
                                                  if (features[15] < -0.221454f) {
                                                      if (features[11] < 0.520833f) {
                                                          return 1;
                                                      } else {
                                                          return 1;
                                                      }
                                                  } else {
                                                      return 1;
                                                  }
                                              } else {
                                                  return 1;
                                              }
                                          } else {
                                              return 1;
                                          }
                                      }
                                  }
                              }
                          } else {
                              return 1;
                          }
                      } else {
                          if (features[6] < 0.070895f) {
                              if (features[4] < 1.742763f) {
                                  if (features[5] < -0.006505f) {
                                      if (features[13] < 0.088802f) {
                                          return 0;
                                      } else {
                                          return 0;
                                      }
                                  } else {
                                      return 2;
                                  }
                              } else {
                                  if (features[1] < 0.245000f) {
                                      return 1;
                                  } else {
                                      return 1;
                                  }
                              }
                          } else {
                              if (features[2] < -0.484128f) {
                                  return 0;
                              } else {
                                  return 3;
                              }
                          }
                      }
                  } else {
                      if (features[0] < 0.018823f) {
                          if (features[15] < -0.691508f) {
                              return 1;
                          } else {
                              if (features[15] < -0.325861f) {
                                  return 2;
                              } else {
                                  return 0;
                              }
                          }
                      } else {
                          if (features[8] < 0.367704f) {
                              if (features[4] < 0.851557f) {
                                  return 1;
                              } else {
                                  if (features[7] < 0.316451f) {
                                      if (features[3] < 1.029128f) {
                                          if (features[11] < 0.817130f) {
                                              if (features[6] < 0.072669f) {
                                                  return 2;
                                              } else {
                                                  if (features[14] < 0.155476f) {
                                                      return 1;
                                                  } else {
                                                      return 2;
                                                  }
                                              }
                                          } else {
                                              if (features[2] < -1.202355f) {
                                                  return 2;
                                              } else {
                                                  return 2;
                                              }
                                          }
                                      } else {
                                          return 1;
                                      }
                                  } else {
                                      return 1;
                                  }
                              }
                          } else {
                              if (features[3] < 1.051848f) {
                                  return 2;
                              } else {
                                  if (features[9] < -0.137346f) {
                                      if (features[10] < 0.316573f) {
                                          return 1;
                                      } else {
                                          return 0;
                                      }
                                  } else {
                                      return 0;
                                  }
                              }
                          }
                      }
                  }
              }
          } else {
              if (features[15] < -0.264439f) {
                  if (features[9] < -0.445484f) {
                      if (features[1] < 0.407974f) {
                          if (features[0] < -0.062400f) {
                              return 2;
                          } else {
                              return 2;
                          }
                      } else {
                          return 1;
                      }
                  } else {
                      if (features[6] < 0.048777f) {
                          if (features[8] < 0.240668f) {
                              if (features[10] < 0.076220f) {
                                  return 0;
                              } else {
                                  return 1;
                              }
                          } else {
                              return 1;
                          }
                      } else {
                          if (features[8] < 0.113992f) {
                              return 0;
                          } else {
                              if (features[9] < -0.132050f) {
                                  if (features[4] < 1.550326f) {
                                      return 1;
                                  } else {
                                      if (features[2] < -1.163693f) {
                                          return 1;
                                      } else {
                                          if (features[0] < 0.081298f) {
                                              if (features[15] < -0.344499f) {
                                                  if (features[12] < 0.018918f) {
                                                      return 1;
                                                  } else {
                                                      return 1;
                                                  }
                                              } else {
                                                  return 1;
                                              }
                                          } else {
                                              if (features[11] < 0.622685f) {
                                                  return 1;
                                              } else {
                                                  return 1;
                                              }
                                          }
                                      }
                                  }
                              } else {
                                  return 1;
                              }
                          }
                      }
                  }
              } else {
                  if (features[13] < -0.154605f) {
                      if (features[6] < 0.106388f) {
                          if (features[8] < 0.158221f) {
                              if (features[8] < 0.140424f) {
                                  if (features[12] < 0.034647f) {
                                      return 0;
                                  } else {
                                      return 0;
                                  }
                              } else {
                                  if (features[7] < 0.145402f) {
                                      return 0;
                                  } else {
                                      return 3;
                                  }
                              }
                          } else {
                              if (features[15] < -0.009594f) {
                                  if (features[9] < -0.175929f) {
                                      if (features[12] < -0.087734f) {
                                          return 0;
                                      } else {
                                          return 0;
                                      }
                                  } else {
                                      if (features[7] < 0.134161f) {
                                          return 0;
                                      } else {
                                          return 1;
                                      }
                                  }
                              } else {
                                  return 0;
                              }
                          }
                      } else {
                          if (features[14] < -0.406337f) {
                              if (features[2] < -0.966020f) {
                                  if (features[3] < 1.920692f) {
                                      return 3;
                                  } else {
                                      return 0;
                                  }
                              } else {
                                  if (features[15] < 0.010650f) {
                                      if (features[1] < 0.418098f) {
                                          if (features[15] < -0.058594f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          return 3;
                                      }
                                  } else {
                                      return 0;
                                  }
                              }
                          } else {
                              if (features[9] < -0.834028f) {
                                  return 2;
                              } else {
                                  if (features[11] < 0.511574f) {
                                      if (features[7] < 0.284912f) {
                                          return 0;
                                      } else {
                                          return 3;
                                      }
                                  } else {
                                      if (features[2] < -0.992749f) {
                                          if (features[2] < -1.237904f) {
                                              return 0;
                                          } else {
                                              return 0;
                                          }
                                      } else {
                                          if (features[7] < 0.407237f) {
                                              if (features[5] < -0.009594f) {
                                                  if (features[7] < 0.207993f) {
                                                      if (features[10] < 0.142815f) {
                                                          return 3;
                                                      } else {
                                                          return 3;
                                                      }
                                                  } else {
                                                      if (features[13] < -0.219752f) {
                                                          return 3;
                                                      } else {
                                                          return 3;
                                                      }
                                                  }
                                              } else {
                                                  return 3;
                                              }
                                          } else {
                                              return 3;
                                          }
                                      }
                                  }
                              }
                          }
                      }
                  } else {
                      if (features[9] < -0.672823f) {
                          if (features[3] < 1.302561f) {
                              if (features[0] < -0.104977f) {
                                  return 2;
                              } else {
                                  if (features[5] < -0.019412f) {
                                      if (features[0] < -0.077360f) {
                                          return 2;
                                      } else {
                                          return 2;
                                      }
                                  } else {
                                      return 2;
                                  }
                              }
                          } else {
                              if (features[3] < 1.926719f) {
                                  if (features[15] < 0.034023f) {
                                      if (features[2] < -0.900243f) {
                                          if (features[8] < 0.363540f) {
                                              return 2;
                                          } else {
                                              return 3;
                                          }
                                      } else {
                                          return 3;
                                      }
                                  } else {
                                      if (features[4] < 2.424694f) {
                                          return 0;
                                      } else {
                                          return 0;
                                      }
                                  }
                              } else {
                                  if (features[0] < -0.119031f) {
                                      return 0;
                                  } else {
                                      return 0;
                                  }
                              }
                          }
                      } else {
                          if (features[5] < -0.072847f) {
                              if (features[8] < 0.392192f) {
                                  if (features[12] < -0.052134f) {
                                      return 0;
                                  } else {
                                      if (features[9] < -0.280262f) {
                                          return 0;
                                      } else {
                                          return 3;
                                      }
                                  }
                              } else {
                                  if (features[6] < 0.151956f) {
                                      return 0;
                                  } else {
                                      if (features[8] < 0.551857f) {
                                          if (features[1] < 0.422903f) {
                                              return 3;
                                          } else {
                                              return 3;
                                          }
                                      } else {
                                          return 0;
                                      }
                                  }
                              }
                          } else {
                              if (features[10] < 0.424015f) {
                                  if (features[11] < 0.900463f) {
                                      if (features[3] < 1.988921f) {
                                          if (features[14] < 0.142038f) {
                                              if (features[15] < -0.192223f) {
                                                  if (features[8] < 0.120873f) {
                                                      return 2;
                                                  } else {
                                                      if (features[5] < 0.013310f) {
                                                          return 0;
                                                      } else {
                                                          return 3;
                                                      }
                                                  }
                                              } else {
                                                  if (features[13] < -0.100437f) {
                                                      if (features[1] < 0.425264f) {
                                                          return 0;
                                                      } else {
                                                          return 3;
                                                      }
                                                  } else {
                                                      if (features[15] < -0.143930f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              }
                                          } else {
                                              if (features[3] < 1.146991f) {
                                                  if (features[2] < -0.752904f) {
                                                      return 2;
                                                  } else {
                                                      if (features[4] < 1.002588f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              } else {
                                                  if (features[7] < 0.215060f) {
                                                      if (features[14] < 0.205224f) {
                                                          return 1;
                                                      } else {
                                                          return 1;
                                                      }
                                                  } else {
                                                      if (features[2] < -0.746491f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  }
                                              }
                                          }
                                      } else {
                                          if (features[4] < 2.623028f) {
                                              return 3;
                                          } else {
                                              if (features[5] < -0.058459f) {
                                                  if (features[0] < -0.010846f) {
                                                      return 3;
                                                  } else {
                                                      return 0;
                                                  }
                                              } else {
                                                  if (features[13] < 0.288054f) {
                                                      if (features[10] < 0.274757f) {
                                                          return 0;
                                                      } else {
                                                          return 0;
                                                      }
                                                  } else {
                                                      return 0;
                                                  }
                                              }
                                          }
                                      }
                                  } else {
                                      return 2;
                                  }
                              } else {
                                  if (features[5] < 0.049795f) {
                                      if (features[0] < 0.012601f) {
                                          return 2;
                                      } else {
                                          if (features[13] < 0.056748f) {
                                              if (features[14] < -0.042374f) {
                                                  return 0;
                                              } else {
                                                  return 0;
                                              }
                                          } else {
                                              return 0;
                                          }
                                      }
                                  } else {
                                      return 3;
                                  }
                              }
                          }
                      }
                  }
              }
          }
        }
        

int32_t random_forest_predict(const float *features, int32_t features_length) {

        int32_t votes[5] = {0,};
        int32_t _class = -1;

        _class = random_forest_tree_0(features, features_length); votes[_class] += 1;
    _class = random_forest_tree_1(features, features_length); votes[_class] += 1;
    _class = random_forest_tree_2(features, features_length); votes[_class] += 1;
    _class = random_forest_tree_3(features, features_length); votes[_class] += 1;
    _class = random_forest_tree_4(features, features_length); votes[_class] += 1;
    _class = random_forest_tree_5(features, features_length); votes[_class] += 1;
    _class = random_forest_tree_6(features, features_length); votes[_class] += 1;
    _class = random_forest_tree_7(features, features_length); votes[_class] += 1;
    _class = random_forest_tree_8(features, features_length); votes[_class] += 1;
    _class = random_forest_tree_9(features, features_length); votes[_class] += 1;
    
        int32_t most_voted_class = -1;
        int32_t most_voted_votes = 0;
        for (int32_t i=0; i<5; i++) {

            if (votes[i] > most_voted_votes) {
                most_voted_class = i;
                most_voted_votes = votes[i];
            }
        }
        return most_voted_class;
    }
    

int random_forest_predict_proba(const float *features, int32_t features_length, float *out, int out_length) {

        int32_t _class = -1;

        for (int i=0; i<out_length; i++) {
            out[i] = 0.0f;
        }

        _class = random_forest_tree_0(features, features_length); out[_class] += 1.0f;
    _class = random_forest_tree_1(features, features_length); out[_class] += 1.0f;
    _class = random_forest_tree_2(features, features_length); out[_class] += 1.0f;
    _class = random_forest_tree_3(features, features_length); out[_class] += 1.0f;
    _class = random_forest_tree_4(features, features_length); out[_class] += 1.0f;
    _class = random_forest_tree_5(features, features_length); out[_class] += 1.0f;
    _class = random_forest_tree_6(features, features_length); out[_class] += 1.0f;
    _class = random_forest_tree_7(features, features_length); out[_class] += 1.0f;
    _class = random_forest_tree_8(features, features_length); out[_class] += 1.0f;
    _class = random_forest_tree_9(features, features_length); out[_class] += 1.0f;
    
        // compute mean
        for (int i=0; i<out_length; i++) {
            out[i] = out[i] / 10;
        }
        return 0;
    }
    