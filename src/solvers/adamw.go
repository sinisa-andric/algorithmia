package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	adamwLearningRate = 0.001
	adamwBeta1        = 0.9
	adamwBeta2        = 0.999
	adamwEpsilon      = 1e-8
	adamwWeightDecay  = 0.01
	adamwMaxSteps     = 1000
	adamwTolerance    = 1e-6
)

// Adamw minimizuje konfigurisanu benchmark funkciju koristeći AdamW: Adam kod koga je weight decay odvojen od
// gradijentnog člana i primenjuje se direktno na parametre
// problem.Point je početna tačka pretrage
func Adamw(problem models.Problem) (result models.Result, err error) {

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

	learningRate := adamwLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	beta1 := adamwBeta1
	if v, ok := problem.Payload["beta1"].(float64); ok {
		beta1 = v
	}

	beta2 := adamwBeta2
	if v, ok := problem.Payload["beta2"].(float64); ok {
		beta2 = v
	}

	epsilon := adamwEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	weightDecay := adamwWeightDecay
	if v, ok := problem.Payload["weight_decay"].(float64); ok {
		weightDecay = v
	}

	maxSteps := adamwMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := adamwTolerance
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

			mHat := m[i] / (1 - beta1T)
			vHat := v[i] / (1 - beta2T)

			point[i] -= learningRate * (mHat/(math.Sqrt(vHat)+epsilon) + weightDecay*point[i])
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
		Method:     "adamw",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
