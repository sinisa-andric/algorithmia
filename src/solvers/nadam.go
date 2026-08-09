package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	nadamLearningRate = 0.001
	nadamBeta1        = 0.9
	nadamBeta2        = 0.999
	nadamEpsilon      = 1e-8
	nadamMaxSteps     = 1000
	nadamTolerance    = 1e-6
)

// Nadam minimizuje konfigurisanu benchmark funkciju koristeći Nesterov-ubrzani Adam: Nesterov lookahead se ugrađuje
// direktno u korekciju prvog momenta pre ažuriranja pozicije
// problem.Point je početna tačka pretrage
func Nadam(problem models.Problem) (result models.Result, err error) {

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

	learningRate := nadamLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	beta1 := nadamBeta1
	if v, ok := problem.Payload["beta1"].(float64); ok {
		beta1 = v
	}

	beta2 := nadamBeta2
	if v, ok := problem.Payload["beta2"].(float64); ok {
		beta2 = v
	}

	epsilon := nadamEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	maxSteps := nadamMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := nadamTolerance
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

		for i := range point {
			m[i] = beta1*m[i] + (1-beta1)*gradient[i]
			v[i] = beta2*v[i] + (1-beta2)*gradient[i]*gradient[i]

			mHat := (beta1*m[i] + (1-beta1)*gradient[i]) / (1 - beta1T)
			vHat := v[i] / (1 - beta2T)

			point[i] -= learningRate * mHat / (math.Sqrt(vHat) + epsilon)
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
		Method:     "nadam",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
