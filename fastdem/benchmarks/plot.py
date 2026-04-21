  import pandas as pd                                                                                                               
  import matplotlib.pyplot as plt                                                                                                   
                  
  # Convergence curves
  df = pd.read_csv("convergence_rmse.csv")
  flat = df[(df.terrain == "Flat") & (df.sigma == 0.05)]                                                                            
  for _, row in flat.iterrows():
      scans = range(1, 101)                                                                                                         
      rmse  = [row[f"scan{i}"] for i in scans]
      plt.plot(scans, rmse, label=row.estimator)                                                                                    
  plt.xlabel("Scan count"); plt.ylabel("RMSE (m)"); plt.legend(); plt.show()
                                                                                                                                    
  # Dynamic response
  df2 = pd.read_csv("dynamic_response.csv")                                                                                         
  for col in ["Kalman_rmse", "P2Quantile_rmse", "StatMean_rmse", "MovingAvg_rmse"]:
      plt.plot(df2.scan, df2[col], label=col.replace("_rmse", ""))                                                                  
  plt.axvline(50, linestyle="--", color="gray", label="Step change")                                                                
  plt.xlabel("Scan"); plt.ylabel("RMSE vs current target (m)"); plt.legend(); plt.show()
