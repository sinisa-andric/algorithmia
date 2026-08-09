package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math/rand/v2"
)

const (
	particleSwarmParticles       = 10
	particleSwarmMaxSteps        = 1000
	particleSwarmTolerance       = 1e-6
	particleSwarmInertia         = 0.7
	particleSwarmCognitiveWeight = 1.5
	particleSwarmSocialWeight    = 1.5
	particleSwarmSpread          = 5.0
)

// ParticleSwarm minimizuje sphere funkciju koristeći optimizaciju rojem čestica: brzina svake čestice spaja inerciju,
// privlačenje ka sopstvenom najboljem rezultatu i privlačenje ka globalno najboljem
// problem.Point inicijalizuje roj i određuje njegovu dimenzionalnost
func ParticleSwarm(problem models.Problem) (result models.Result, err error) {

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

	particles := particleSwarmParticles
	if v, ok := problem.Payload["particles"].(float64); ok {
		particles = int(v)
	}

	maxSteps := particleSwarmMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := particleSwarmTolerance
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
		for i := range position {
			position[i] = problem.Point[i] + (rand.Float64()*2-1)*particleSwarmSpread
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

			for i := 0; i < dimensions; i++ {
				velocity[i] = particleSwarmInertia*velocity[i] +
					particleSwarmCognitiveWeight*rand.Float64()*(personalBest[p][i]-position[i]) +
					particleSwarmSocialWeight*rand.Float64()*(globalBest[i]-position[i])
				position[i] += velocity[i]
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
		Method:     "particle_swarm",
		Point:      globalBest,
		Value:      globalBestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
