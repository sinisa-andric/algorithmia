package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	selfAdaptiveESPopulation = 10
	selfAdaptiveESMaxSteps   = 1000
	selfAdaptiveESTolerance  = 1e-6
	selfAdaptiveESInitSigma  = 1.0
)

// SelfAdaptiveES minimizuje sphere funkciju koristeći samo-adaptivnu strategiju evolucije: svaka jedinka nosi
// sopstvenu veličinu koraka mutacije (sigma) koja se log-normalno samo-adaptira uz njenu poziciju i ažurira se samo
// kad mutacija poboljša rezultat
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func SelfAdaptiveES(problem models.Problem) (result models.Result, err error) {

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

	population := selfAdaptiveESPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := selfAdaptiveESMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := selfAdaptiveESTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	tau := 1 / math.Sqrt(float64(dimensions))

	positions := make([][]float64, population)
	sigmas := make([]float64, population)
	for i := range positions {
		position := make([]float64, dimensions)
		for d := range position {
			position[d] = problem.Point[d] + randNorm()*selfAdaptiveESInitSigma
		}
		positions[i] = position
		sigmas[i] = selfAdaptiveESInitSigma
	}

	best := append([]float64(nil), positions[0]...)
	bestValue := fn.Evaluate(best)
	for _, position := range positions {
		if value := fn.Evaluate(position); value < bestValue {
			bestValue = value
			best = append([]float64(nil), position...)
		}
	}

	steps := 0
	for ; steps < maxSteps && bestValue >= tolerance; steps++ {

		for i := range positions {
			newSigma := sigmas[i] * math.Exp(tau*randNorm())

			newPosition := make([]float64, dimensions)
			for d := range newPosition {
				newPosition[d] = positions[i][d] + newSigma*randNorm()
			}

			if fn.Evaluate(newPosition) < fn.Evaluate(positions[i]) {
				positions[i] = newPosition
				sigmas[i] = newSigma

				if value := fn.Evaluate(newPosition); value < bestValue {
					bestValue = value
					best = append([]float64(nil), newPosition...)
				}
				if includeTrajectory {
					trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
				}
			}
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "self_adaptive_es",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
