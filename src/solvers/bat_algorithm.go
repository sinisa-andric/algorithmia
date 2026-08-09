package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math/rand/v2"
)

const (
	batAlgorithmBats      = 20
	batAlgorithmMaxSteps  = 500
	batAlgorithmFMin      = 0.0
	batAlgorithmFMax      = 2.0
	batAlgorithmLoudness  = 0.5
	batAlgorithmPulseRate = 0.5
	batAlgorithmTolerance = 1e-6
	batAlgorithmSpread    = 10.0
)

// BatAlgorithm minimizuje sphere funkciju koristeći bat algoritam: frekvencija eholokacije usmerava brzinu svakog
// šišmiša ka trenutno najboljem, uz povremenu lokalnu eksploraciju nasumičnom šetnjom oko njega
// problem.Point inicijalizuje koloniju i određuje njenu dimenzionalnost
func BatAlgorithm(problem models.Problem) (result models.Result, err error) {

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

	batCount := batAlgorithmBats
	if v, ok := problem.Payload["bats"].(float64); ok {
		batCount = int(v)
	}

	maxSteps := batAlgorithmMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	fMin := batAlgorithmFMin
	if v, ok := problem.Payload["f_min"].(float64); ok {
		fMin = v
	}

	fMax := batAlgorithmFMax
	if v, ok := problem.Payload["f_max"].(float64); ok {
		fMax = v
	}

	loudness := batAlgorithmLoudness
	if v, ok := problem.Payload["loudness"].(float64); ok {
		loudness = v
	}

	pulseRate := batAlgorithmPulseRate
	if v, ok := problem.Payload["pulse_rate"].(float64); ok {
		pulseRate = v
	}

	tolerance := batAlgorithmTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	bats := make([][]float64, batCount)
	velocities := make([][]float64, batCount)
	for i := range bats {
		bat := make([]float64, dimensions)
		for d := range bat {
			bat[d] = problem.Point[d] + (rand.Float64()*2-1)*batAlgorithmSpread
		}
		bats[i] = bat
		velocities[i] = make([]float64, dimensions)
	}

	best := append([]float64(nil), bats[0]...)
	bestValue := fn.Evaluate(best)
	for _, bat := range bats {
		if value := fn.Evaluate(bat); value < bestValue {
			bestValue = value
			best = append([]float64(nil), bat...)
		}
	}

	steps := 0
	for ; steps < maxSteps && bestValue >= tolerance; steps++ {

		for i, bat := range bats {
			velocity := velocities[i]
			freq := fMin + (fMax-fMin)*rand.Float64()

			newPos := make([]float64, dimensions)
			for d := range bat {
				velocity[d] += (bat[d] - best[d]) * freq
				newPos[d] = bat[d] + velocity[d]
			}

			if rand.Float64() > pulseRate {
				for d := range newPos {
					newPos[d] = best[d] + 0.01*loudness*randNorm()
				}
			}

			if fn.Evaluate(newPos) < fn.Evaluate(bat) && rand.Float64() < loudness {
				bats[i] = newPos

				if value := fn.Evaluate(newPos); value < bestValue {
					bestValue = value
					best = append([]float64(nil), newPos...)
				}
			}
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "bat_algorithm",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
