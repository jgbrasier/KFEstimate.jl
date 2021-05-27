Random.seed!(0)

# Linear Dynamic model
# = x = [p; v]
dt = 0.01
A = [1.0 dt; 0.0 1.0]
B = zeros(2, 2)
W = 0.01*Matrix{Float64}(I, 2, 2)

dyn = LinearDynamicModel(A, B, W)

# Linear Observation Model
# observe postition
H = [1 0]
obs = LinearObservationModel(H)

# define linear kalman filter
kf = KalmanFilter(dyn, obs)

# simulation
R_gt = 0.5*Matrix{Float64}(I, 1, 1) # ground truth noise covariance
time_step = 0.1:dt:1000
s0 = State([0.0, 1.0], Matrix{Float64}(I, 2, 2)) # initial state belief
