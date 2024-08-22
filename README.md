# PP-PINN
 Physics-Informed Neural Network (PINN) is a deep learning framework that
 has been widely employed to solve spatial-temporal partial di erential equa
tions (PDEs) across various elds. However, recent numerical experiments
 indicate that the vanilla-PINN often struggles with PDEs featuring high
frequency solutions or strong nonlinearity. To enhance PINNs performance,
 we propose a novel strategy called the Preconditioning-Pretraining Physics
Informed Neural Network (PP-PINN). This approach involves transforming
 the original task into a new system characterized by low frequency and weak
 nonlinearity over an extended time scale. The transformed PDEs are then
 solved using a pretraining approach. Additionally, we introduce a new con
straint termed xed point , which is bene cial for scenarios with extremely
 high frequency or strong nonlinearity. To demonstrate the e cacy of our
 method, we apply the newly developed strategy to three di erent equations,achieving improved accuracy and reduced computational costs compared to
 previous approaches. The e ectiveness and interpretability of our PP-PINN
 are also discussed.
