package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	pidOptimizerLearningRate = 0.1
	pidOptimizerKp           = 1.0
	pidOptimizerKi           = 0.01
	pidOptimizerKd           = 10.0
	pidOptimizerMaxSteps     = 1000
	pidOptimizerTolerance    = 1e-6
	pidOptimizerStepClip     = 1.0
)

// PidOptimizer minimizuje konfigurisanu benchmark funkciju koristeći kontrolno-teorijski PID pristup: korak je
// zbir proporcionalnog (gradijent), integralnog (akumulirani gradijent) i derivacionog (promena gradijenta) člana
// — potpuno drugačija filozofija od EMA/momentum pristupa koji dominiraju ovim modulom
// problem.Point je početna tačka pretrage
func PidOptimizer(problem models.Problem) (result models.Result, err error) {

	if len(problem.Point) == 0 {
		err = fmt.Errorf("starting point is required")
		return result, err
	}

	fnName, _ := problem.Payload["function"].(string)
	fn, err := functions.Get(fnName)
	if err != nil {
		return result, err
	}
	if err := fn.ValidateDimension(len(problem.Point)); err != nil {
		return result, err
	}

	learningRate := pidOptimizerLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	kp := pidOptimizerKp
	if v, ok := problem.Payload["kp"].(float64); ok {
		kp = v
	}

	ki := pidOptimizerKi
	if v, ok := problem.Payload["ki"].(float64); ok {
		ki = v
	}

	kd := pidOptimizerKd
	if v, ok := problem.Payload["kd"].(float64); ok {
		kd = v
	}

	maxSteps := pidOptimizerMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := pidOptimizerTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x := append([]float64(nil), problem.Point...)
	integral := make([]float64, dimensions)
	gradPrev := make([]float64, dimensions)

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		delta := make([]float64, dimensions)
		for d := range delta {
			integral[d] += grad[d]
			// na prvom koraku gradPrev je izmišljeni placeholder (nula), pa bi derivative=grad[d]-0 bio
			// pun gradijent, ne stvarna promena — sa kd=10.0 taj lažni "udarac" gura x u stabilan
			// dvo-koračni ciklus (klipovan na ±1) koji troši stotine koraka pre nego što se raspadne
			var derivative float64
			if steps > 0 {
				derivative = grad[d] - gradPrev[d]
			}
			delta[d] = -learningRate * (kp*grad[d] + ki*integral[d] + kd*derivative)
		}
		delta = clipStep(delta, pidOptimizerStepClip)
		for d := range x {
			x[d] += delta[d]
		}
		gradPrev = grad

		value := fn.Evaluate(x)
		if value < bestValue {
			bestValue = value
			best = append([]float64(nil), x...)
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
		}

		if math.Abs(bestValue-prevBestValue) < tolerance {
			noImprove++
		} else {
			noImprove = 0
		}
		if noImprove >= maxNoImprove {
			steps++
			break
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, fn.Evaluate(best), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "pid_optimizer",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
