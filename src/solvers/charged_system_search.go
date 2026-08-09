package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	chargedSystemSearchAgents              = 20
	chargedSystemSearchMaxSteps            = 500
	chargedSystemSearchTolerance           = 1e-6
	chargedSystemSearchSpread              = 10.0
	chargedSystemSearchAttractionWeight    = 0.5
	chargedSystemSearchVelocityInertia     = 0.5
	chargedSystemSearchVelocityForceWeight = 0.1
	chargedSystemSearchVelocityBound       = 1.0
	chargedSystemSearchPositionBound       = 15.0
	chargedSystemSearchEpsilon             = 1e-10
)

// ChargedSystemSearch minimizuje sphere funkciju koristeći Charged System Search: svaki agent se gura ka boljim
// (niže-energetskim) susedima elektrostatičkom silom po zakonu inverznog kvadrata,
// plus privlačenjem ka trenutno najboljem
// problem.Point inicijalizuje roj i određuje njegovu dimenzionalnost
func ChargedSystemSearch(problem models.Problem) (result models.Result, err error) {

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

	agents := chargedSystemSearchAgents
	if v, ok := problem.Payload["agents"].(float64); ok {
		agents = int(v)
	}

	maxSteps := chargedSystemSearchMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := chargedSystemSearchTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	positions := make([][]float64, agents)
	velocities := make([][]float64, agents)
	for i := range positions {
		position := make([]float64, dimensions)
		for d := range position {
			position[d] = problem.Point[d] + (rand.Float64()*2-1)*chargedSystemSearchSpread
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

		for i := range positions {
			force := make([]float64, dimensions)
			iValue := fn.Evaluate(positions[i])

			for j := range positions {
				if j == i {
					continue
				}
				if fn.Evaluate(positions[j]) >= iValue {
					continue
				}

				distanceSquared := 0.0
				for d := 0; d < dimensions; d++ {
					diff := positions[j][d] - positions[i][d]
					distanceSquared += diff * diff
				}
				distance := math.Sqrt(distanceSquared)

				for d := 0; d < dimensions; d++ {
					force[d] += (positions[j][d] - positions[i][d]) / (distance + chargedSystemSearchEpsilon)
				}
			}

			for d := 0; d < dimensions; d++ {
				force[d] += chargedSystemSearchAttractionWeight * (best[d] - positions[i][d])
			}

			velocity := velocities[i]
			for d := 0; d < dimensions; d++ {
				velocity[d] = chargedSystemSearchVelocityInertia*velocity[d] +
					chargedSystemSearchVelocityForceWeight*rand.Float64()*force[d]

				if velocity[d] > chargedSystemSearchVelocityBound {
					velocity[d] = chargedSystemSearchVelocityBound
				} else if velocity[d] < -chargedSystemSearchVelocityBound {
					velocity[d] = -chargedSystemSearchVelocityBound
				}

				positions[i][d] += velocity[d]

				if positions[i][d] > chargedSystemSearchPositionBound {
					positions[i][d] = chargedSystemSearchPositionBound
				} else if positions[i][d] < -chargedSystemSearchPositionBound {
					positions[i][d] = -chargedSystemSearchPositionBound
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
		Method:     "charged_system_search",
		Point:      globalBest,
		Value:      globalBestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
