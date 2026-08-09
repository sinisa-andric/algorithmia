package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

func randNorm() float64 {
	u1 := 1.0 - rand.Float64()
	u2 := 1.0 - rand.Float64()
	return math.Sqrt(-2.0*math.Log(u1)) * math.Cos(2.0*math.Pi*u2)
}

// CmaEs minimizuje konfigurisanu benchmark funkciju koristeći pojednostavljeni CMA-ES:
// uzorkuje populaciju iz izotropne Gausove raspodele oko tekuće sredine, novu sredinu računa kao prosek elitnog
// podskupa, a sigma se skalira gore ili dole u zavisnosti od toga da li se rešenje poboljšalo
// problem.Point je početna sredina raspodele iz koje se uzorkuje populacija
func CmaEs(problem models.Problem) (models.Result, error) {
	if len(problem.Point) == 0 {
		return models.Result{}, fmt.Errorf("starting point is required")
	}

	fnName, _ := problem.Payload["function"].(string)
	fn, err := functions.Get(fnName)
	if err != nil {
		return models.Result{}, err
	}
	if err := fn.ValidateDimension(len(problem.Point)); err != nil {
		return models.Result{}, err
	}

	n := len(problem.Point)
	maxSteps := 1000
	tolerance := 1e-6
	popSize := 10

	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	if v, ok := problem.Payload["population"].(float64); ok {
		popSize = int(v)
	}

	mean := make([]float64, n)
	copy(mean, problem.Point)
	sigma := 1.0
	if v, ok := problem.Payload["sigma"].(float64); ok {
		sigma = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	bestPoint := make([]float64, n)
	copy(bestPoint, mean)
	bestValue := fn.Evaluate(mean)
	steps := 0

	for steps < maxSteps && sigma > tolerance {
		steps++

		// generiši population uzoraka
		samples := make([][]float64, popSize)
		values := make([]float64, popSize)
		for i := range samples {
			samples[i] = make([]float64, n)
			for j := range samples[i] {
				samples[i][j] = mean[j] + sigma*randNorm()
			}
			values[i] = fn.Evaluate(samples[i])
			if values[i] < bestValue {
				bestValue = values[i]
				copy(bestPoint, samples[i])
			}
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, bestPoint, bestValue, false)
		}

		// sortiraj po vrednosti
		for i := 0; i < popSize-1; i++ {
			for j := i + 1; j < popSize; j++ {
				if values[j] < values[i] {
					values[i], values[j] = values[j], values[i]
					samples[i], samples[j] = samples[j], samples[i]
				}
			}
		}

		// novi mean = prosek prvih popSize/2 (elite)
		elite := popSize / 2
		newMean := make([]float64, n)
		for i := 0; i < elite; i++ {
			for j := range newMean {
				newMean[j] += samples[i][j]
			}
		}
		for j := range newMean {
			newMean[j] /= float64(elite)
		}

		// adaptiraj sigma na osnovu uspeha
		if fn.Evaluate(newMean) < fn.Evaluate(mean) {
			sigma *= 0.99
		} else {
			sigma *= 1.2
		}
		mean = newMean
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, bestPoint, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	return models.Result{
		Method:     "cma_es",
		Point:      bestPoint,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}, nil
}
