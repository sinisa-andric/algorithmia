package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	salpSwarmSalps     = 20
	salpSwarmMaxSteps  = 500
	salpSwarmTolerance = 1e-6
	salpSwarmRange     = 15.0
)

// SalpSwarm minimizuje sphere funkciju koristeći Salp Swarm Algorithm: vodeći salpi se kreću ka trenutno najboljem sa
// koeficijentom koji eksponencijalno opada tokom izvršavanja, a salpi sledbenici se kreću ka prethodnom salpu u lancu
// problem.Point samo određuje dimenzionalnost
func SalpSwarm(problem models.Problem) (result models.Result, err error) {

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

	count := salpSwarmSalps
	if v, ok := problem.Payload["salps"].(float64); ok {
		count = int(v)
	}
	if count < 2 {
		count = 2
	}

	maxSteps := salpSwarmMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := salpSwarmTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	searchRange := salpSwarmRange
	if v, ok := problem.Payload["range"].(float64); ok {
		searchRange = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	swarm := make([][]float64, count)
	for i := range swarm {
		position := make([]float64, dimensions)
		for d := range position {
			position[d] = (rand.Float64()*2 - 1) * searchRange
		}
		swarm[i] = position
	}

	best := append([]float64(nil), swarm[0]...)
	bestValue := fn.Evaluate(best)
	for _, position := range swarm {
		if value := fn.Evaluate(position); value < bestValue {
			bestValue = value
			best = append([]float64(nil), position...)
		}
	}

	leaders := count / 2

	steps := 0
	for ; steps < maxSteps && bestValue >= tolerance; steps++ {

		c1 := 2 * math.Exp(-math.Pow(4*float64(steps)/float64(maxSteps), 2))

		for i := 0; i < leaders; i++ {
			for d := 0; d < dimensions; d++ {
				term := c1 * ((searchRange-(-searchRange))*rand.Float64() + (-searchRange))
				if rand.Float64() >= 0.5 {
					swarm[i][d] = best[d] + term
				} else {
					swarm[i][d] = best[d] - term
				}
			}
		}

		for i := leaders; i < count; i++ {
			for d := 0; d < dimensions; d++ {
				swarm[i][d] = 0.5 * (swarm[i][d] + swarm[i-1][d])
			}
		}

		for _, position := range swarm {
			if value := fn.Evaluate(position); value < bestValue {
				bestValue = value
				best = append([]float64(nil), position...)
			}
			if includeTrajectory {
				trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
			}
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "salp_swarm",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
