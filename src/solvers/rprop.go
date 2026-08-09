package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
)

const (
	rpropEtaPlus     = 1.2
	rpropEtaMinus    = 0.5
	rpropStepMin     = 1e-6
	rpropStepMax     = 50.0
	rpropMaxSteps    = 1000
	rpropTolerance   = 1e-6
	rpropInitialStep = 0.1
)

// Rprop minimizuje konfigurisanu benchmark funkciju koristeći Resilient Backpropagation: veličina koraka po dimenziji
// raste ili opada isključivo na osnovu znaka uzastopnih gradijenata, nikad na osnovu njihove magnitude
// problem.Point je početna tačka pretrage
func Rprop(problem models.Problem) (result models.Result, err error) {

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

	etaPlus := rpropEtaPlus
	if v, ok := problem.Payload["eta_plus"].(float64); ok {
		etaPlus = v
	}

	etaMinus := rpropEtaMinus
	if v, ok := problem.Payload["eta_minus"].(float64); ok {
		etaMinus = v
	}

	stepMin := rpropStepMin
	if v, ok := problem.Payload["step_min"].(float64); ok {
		stepMin = v
	}

	stepMax := rpropStepMax
	if v, ok := problem.Payload["step_max"].(float64); ok {
		stepMax = v
	}

	maxSteps := rpropMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := rpropTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	delta := make([]float64, len(point))
	gradPrev := make([]float64, len(point))
	for i := range delta {
		delta[i] = rpropInitialStep
	}
	steps := 0

	for ; steps < maxSteps; steps++ {
		gradient := fn.Gradient(point)
		if norm(gradient) < tolerance {
			break
		}

		for i := range point {
			product := gradient[i] * gradPrev[i]
			switch {
			case product > 0:
				delta[i] = min(delta[i]*etaPlus, stepMax)
			case product < 0:
				delta[i] = max(delta[i]*etaMinus, stepMin)
			}
			point[i] -= sign(gradient[i]) * delta[i]
		}

		gradPrev = gradient

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "rprop",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}

func sign(x float64) float64 {
	switch {
	case x > 0:
		return 1
	case x < 0:
		return -1
	default:
		return 0
	}
}
