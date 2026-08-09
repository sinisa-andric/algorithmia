package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math/rand/v2"
)

const (
	psoGDParticles       = 20
	psoGDLocalSteps      = 5
	psoGDMaxSteps        = 500
	psoGDTolerance       = 1e-6
	psoGDSpread          = 10.0
	psoGDInertia         = 0.7
	psoGDCognitiveWeight = 1.5
	psoGDSocialWeight    = 1.5
	psoGDLocalLearnRate  = 0.01
)

// PSOGD minimizuje sphere funkciju koristeći hibrid optimizacije rojem čestica i lokalnog gradijentnog spusta: svaki
// korak izvršava uobičajeno PSO ažuriranje brzine i pozicije, a zatim dorađuje svaku česticu sa nekoliko koraka
// gradijentnog spusta
// problem.Point inicijalizuje roj i određuje njegovu dimenzionalnost
func PSOGD(problem models.Problem) (result models.Result, err error) {

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

	particles := psoGDParticles
	if v, ok := problem.Payload["particles"].(float64); ok {
		particles = int(v)
	}

	localSteps := psoGDLocalSteps
	if v, ok := problem.Payload["local_steps"].(float64); ok {
		localSteps = int(v)
	}

	maxSteps := psoGDMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := psoGDTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	positions := make([][]float64, particles)
	velocities := make([][]float64, particles)
	personalBest := make([][]float64, particles)
	personalBestValue := make([]float64, particles)

	globalBest := append([]float64(nil), problem.Point...)
	globalBestValue := fn.Evaluate(globalBest)

	for p := 0; p < particles; p++ {
		position := make([]float64, dimensions)
		for d := range position {
			position[d] = problem.Point[d] + (rand.Float64()*2-1)*psoGDSpread
		}

		positions[p] = position
		velocities[p] = make([]float64, dimensions)
		personalBest[p] = append([]float64(nil), position...)
		personalBestValue[p] = fn.Evaluate(position)

		if personalBestValue[p] < globalBestValue {
			globalBestValue = personalBestValue[p]
			globalBest = append([]float64(nil), position...)
		}
	}

	steps := 0
	for ; steps < maxSteps && globalBestValue >= tolerance; steps++ {

		for p := 0; p < particles; p++ {
			position := positions[p]
			velocity := velocities[p]

			for d := 0; d < dimensions; d++ {
				velocity[d] = psoGDInertia*velocity[d] +
					psoGDCognitiveWeight*rand.Float64()*(personalBest[p][d]-position[d]) +
					psoGDSocialWeight*rand.Float64()*(globalBest[d]-position[d])
				position[d] += velocity[d]
			}

			for s := 0; s < localSteps; s++ {
				gradient := fn.Gradient(position)
				for d := range position {
					position[d] -= psoGDLocalLearnRate * gradient[d]
				}
			}

			value := fn.Evaluate(position)
			if value < personalBestValue[p] {
				personalBestValue[p] = value
				personalBest[p] = append([]float64(nil), position...)
			}
			if value < globalBestValue {
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
		Method:     "pso_gd",
		Point:      globalBest,
		Value:      globalBestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
