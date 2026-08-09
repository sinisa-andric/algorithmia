package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	radamLearningRate = 0.001
	radamBeta1        = 0.9
	radamBeta2        = 0.999
	radamEpsilon      = 1e-8
	radamMaxSteps     = 1000
	radamTolerance    = 1e-6
)

// Radam minimizuje konfigurisanu benchmark funkciju koristeći Rectified Adam: kada procena varijanse gradijenta još
// nije pouzdana algoritam koristi obični SGD korigovan momentumom, a inače puni Adam korak
// problem.Point je početna tačka pretrage
func Radam(problem models.Problem) (result models.Result, err error) {

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

	learningRate := radamLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	beta1 := radamBeta1
	if v, ok := problem.Payload["beta1"].(float64); ok {
		beta1 = v
	}

	beta2 := radamBeta2
	if v, ok := problem.Payload["beta2"].(float64); ok {
		beta2 = v
	}

	epsilon := radamEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	maxSteps := radamMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := radamTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	rhoInf := 2/(1-beta2) - 1

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
		rhoT := rhoInf - 2*t*beta2T/(1-beta2T)

		for i := range point {
			m[i] = beta1*m[i] + (1-beta1)*gradient[i]
			v[i] = beta2*v[i] + (1-beta2)*gradient[i]*gradient[i]

			mHat := m[i] / (1 - beta1T)

			if rhoT > 4 {
				vHat := math.Sqrt(v[i] / (1 - beta2T))
				r := math.Sqrt((rhoT - 4) * (rhoT - 2) * rhoInf / ((rhoInf - 4) * (rhoInf - 2) * rhoT))
				point[i] -= learningRate * r * mHat / (vHat + epsilon)
			} else {
				point[i] -= learningRate * mHat
			}
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
		Method:     "radam",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
