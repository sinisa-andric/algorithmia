package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	adadeltaRho       = 0.95
	adadeltaEpsilon   = 1e-6
	adadeltaMaxSteps  = 1000
	adadeltaTolerance = 1e-6
)

// Adadelta minimizuje konfigurisanu benchmark funkciju koristeći AdaDelta: adaptivni metod bez eksplicitnog learning
// rate-a, koji korak određuje iz odnosa tekućih prosečnih kvadrata gradijenata i prethodnih ažuriranja
// problem.Point je početna tačka pretrage
func Adadelta(problem models.Problem) (result models.Result, err error) {

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

	rho := adadeltaRho
	if v, ok := problem.Payload["rho"].(float64); ok {
		rho = v
	}

	epsilon := adadeltaEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	maxSteps := adadeltaMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := adadeltaTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	meanGradSq := make([]float64, len(point))
	meanUpdateSq := make([]float64, len(point))
	steps := 0

	for ; steps < maxSteps; steps++ {
		gradient := fn.Gradient(point)
		if norm(gradient) < tolerance {
			break
		}
		for i := range point {
			meanGradSq[i] = rho*meanGradSq[i] + (1-rho)*gradient[i]*gradient[i]

			delta := -math.Sqrt(meanUpdateSq[i]+epsilon) / math.Sqrt(meanGradSq[i]+epsilon) * gradient[i]

			meanUpdateSq[i] = rho*meanUpdateSq[i] + (1-rho)*delta*delta
			point[i] += delta
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
		Method:     "adadelta",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
