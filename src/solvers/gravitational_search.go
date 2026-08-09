package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	gravitationalSearchAgents        = 20
	gravitationalSearchMaxSteps      = 500
	gravitationalSearchTolerance     = 1e-6
	gravitationalSearchSpread        = 10.0
	gravitationalSearchG0            = 100.0
	gravitationalSearchGDecay        = 20.0
	gravitationalSearchVelocityBound = 2.0
	gravitationalSearchPositionBound = 15.0
	gravitationalSearchEpsilon       = 1e-10
)

// GravitationalSearch minimizuje sphere funkciju koristeći Gravitational Search Algorithm: teži (bolji) agenti
// privlače ostale preko gravitacione konstante koja eksponencijalno opada tokom izvršavanja
// problem.Point inicijalizuje roj i određuje njegovu dimenzionalnost
func GravitationalSearch(problem models.Problem) (result models.Result, err error) {

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

	agents := gravitationalSearchAgents
	if v, ok := problem.Payload["agents"].(float64); ok {
		agents = int(v)
	}

	maxSteps := gravitationalSearchMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := gravitationalSearchTolerance
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
			position[d] = problem.Point[d] + (rand.Float64()*2-1)*gravitationalSearchSpread
		}
		positions[i] = position
		velocities[i] = make([]float64, dimensions)
	}

	globalBest := append([]float64(nil), positions[0]...)
	globalBestValue := fn.Evaluate(globalBest)
	for _, position := range positions {
		if value := fn.Evaluate(position); value < globalBestValue {
			globalBestValue = value
			globalBest = append([]float64(nil), position...)
		}
	}

	steps := 0
	for ; steps < maxSteps && globalBestValue >= tolerance; steps++ {

		g := gravitationalSearchG0 * math.Exp(-gravitationalSearchGDecay*float64(steps)/float64(maxSteps))

		masses := make([]float64, agents)
		totalMass := 0.0
		for i, position := range positions {
			masses[i] = 1 / (fn.Evaluate(position) + gravitationalSearchEpsilon)
			totalMass += masses[i]
		}
		for i := range masses {
			masses[i] /= totalMass
		}

		for i := range positions {
			force := make([]float64, dimensions)

			for j := range positions {
				if j == i {
					continue
				}

				distance := 0.0
				for d := 0; d < dimensions; d++ {
					diff := positions[j][d] - positions[i][d]
					distance += diff * diff
				}
				distance = math.Sqrt(distance)

				for d := 0; d < dimensions; d++ {
					force[d] += g * masses[j] * (positions[j][d] - positions[i][d]) / (distance + gravitationalSearchEpsilon)
				}
			}

			velocity := velocities[i]
			for d := 0; d < dimensions; d++ {
				velocity[d] = rand.Float64()*velocity[d] + rand.Float64()*force[d]

				if velocity[d] > gravitationalSearchVelocityBound {
					velocity[d] = gravitationalSearchVelocityBound
				} else if velocity[d] < -gravitationalSearchVelocityBound {
					velocity[d] = -gravitationalSearchVelocityBound
				}

				positions[i][d] += velocity[d]

				if positions[i][d] > gravitationalSearchPositionBound {
					positions[i][d] = gravitationalSearchPositionBound
				} else if positions[i][d] < -gravitationalSearchPositionBound {
					positions[i][d] = -gravitationalSearchPositionBound
				}
			}
		}

		for _, position := range positions {
			if value := fn.Evaluate(position); value < globalBestValue {
				globalBestValue = value
				globalBest = append([]float64(nil), position...)
			}
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, globalBest, globalBestValue, false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, globalBest, globalBestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "gravitational_search",
		Point:      globalBest,
		Value:      globalBestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
