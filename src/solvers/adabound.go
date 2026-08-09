package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	adaboundLearningRate = 0.001
	adaboundBeta1        = 0.9
	adaboundBeta2        = 0.999
	adaboundEpsilon      = 1e-8
	adaboundFinalLr      = 0.1
	adaboundGamma        = 1e-3
	adaboundMaxSteps     = 1000
	adaboundTolerance    = 1e-6
)

// Adabound minimizuje konfigurisanu benchmark funkciju koristeći AdaBound: Adam kod koga je learning rate po koraku
// ograničen dinamičkim granicama koje konvergiraju ka final_lr kako broj koraka raste
// problem.Point je početna tačka pretrage
func Adabound(problem models.Problem) (result models.Result, err error) {

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

	learningRate := adaboundLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	beta1 := adaboundBeta1
	if v, ok := problem.Payload["beta1"].(float64); ok {
		beta1 = v
	}

	beta2 := adaboundBeta2
	if v, ok := problem.Payload["beta2"].(float64); ok {
		beta2 = v
	}

	epsilon := adaboundEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	finalLr := adaboundFinalLr
	if v, ok := problem.Payload["final_lr"].(float64); ok {
		finalLr = v
	}

	gamma := adaboundGamma
	if v, ok := problem.Payload["gamma"].(float64); ok {
		gamma = v
	}

	maxSteps := adaboundMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := adaboundTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	m := make([]float64, len(point))
	v := make([]float64, len(point))
	steps := 0

	for ; steps < maxSteps; steps++ {
		gradient := fn.Gradient(point)
		if norm(gradient) < tolerance {
			break
		}

		t := float64(steps + 1)
		beta1T := math.Pow(beta1, t)
		beta2T := math.Pow(beta2, t)

		etaLower := finalLr * (1 - 1/(gamma*t+1))
		etaUpper := finalLr * (1 + 1/(gamma*t))

		for i := range point {
			m[i] = beta1*m[i] + (1-beta1)*gradient[i]
			v[i] = beta2*v[i] + (1-beta2)*gradient[i]*gradient[i]

			mHat := m[i] / (1 - beta1T)
			vHat := v[i] / (1 - beta2T)

			step := learningRate / (math.Sqrt(vHat) + epsilon)
			step = math.Max(etaLower, math.Min(etaUpper, step))

			point[i] -= step * mHat
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "adabound",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
