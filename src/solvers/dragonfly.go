package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math/rand/v2"
)

const (
	dragonflyCount         = 20
	dragonflyMaxSteps      = 500
	dragonflyTolerance     = 1e-6
	dragonflySpread        = 10.0
	dragonflyInertia       = 0.7
	dragonflyAttraction    = 0.5
	dragonflyRandomWalk    = 0.1
	dragonflyVelocityBound = 2.0
	dragonflyPositionBound = 15.0
)

// Dragonfly minimizuje sphere funkciju koristeći pojednostavljeno ažuriranje inspirisano vilinim konjicima: brzina
// svake jedinke spaja inerciju, privlačenje ka trenutno najboljem i malu nasumičnu šetnju
// problem.Point inicijalizuje roj i određuje njegovu dimenzionalnost
func Dragonfly(problem models.Problem) (result models.Result, err error) {

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

	count := dragonflyCount
	if v, ok := problem.Payload["dragonflies"].(float64); ok {
		count = int(v)
	}

	maxSteps := dragonflyMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := dragonflyTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	positions := make([][]float64, count)
	velocities := make([][]float64, count)
	for i := range positions {
		position := make([]float64, dimensions)
		for d := range position {
			position[d] = problem.Point[d] + (rand.Float64()*2-1)*dragonflySpread
		}
		positions[i] = position
		velocities[i] = make([]float64, dimensions)
	}

	best := append([]float64(nil), positions[0]...)
	bestValue := fn.Evaluate(best)
	for _, position := range positions {
		if value := fn.Evaluate(position); value < bestValue {
			bestValue = value
			best = append([]float64(nil), position...)
		}
	}

	globalBest := append([]float64(nil), best...)
	globalBestValue := bestValue

	steps := 0
	for ; steps < maxSteps && globalBestValue >= tolerance; steps++ {

		for i, position := range positions {
			velocity := velocities[i]

			for d := 0; d < dimensions; d++ {
				attraction := dragonflyAttraction * (best[d] - position[d])
				randomWalk := dragonflyRandomWalk * randNorm()

				velocity[d] = dragonflyInertia*velocity[d] + attraction + randomWalk

				if velocity[d] > dragonflyVelocityBound {
					velocity[d] = dragonflyVelocityBound
				} else if velocity[d] < -dragonflyVelocityBound {
					velocity[d] = -dragonflyVelocityBound
				}

				position[d] += velocity[d]

				if position[d] > dragonflyPositionBound {
					position[d] = dragonflyPositionBound
				} else if position[d] < -dragonflyPositionBound {
					position[d] = -dragonflyPositionBound
				}
			}
		}

		bestValue = fn.Evaluate(positions[0])
		best = append([]float64(nil), positions[0]...)
		for _, position := range positions {
			if value := fn.Evaluate(position); value < bestValue {
				bestValue = value
				best = append([]float64(nil), position...)
			}
			if includeTrajectory {
				trajectory = recordTrajectory(trajectory, steps, globalBest, globalBestValue, false)
			}
		}

		if bestValue < globalBestValue {
			globalBestValue = bestValue
			globalBest = append([]float64(nil), best...)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, globalBest, globalBestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "dragonfly",
		Point:      globalBest,
		Value:      globalBestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
