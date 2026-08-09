package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	secretaryBirdPopulation = 20
	secretaryBirdMaxSteps   = 1000
	secretaryBirdTolerance  = 1e-6
	secretaryBirdRangeLow   = -5.0
	secretaryBirdRangeHigh  = 5.0
)

// SecretaryBird minimizuje konfigurisanu benchmark funkciju koristeći Secretary Bird Optimization Algorithm: u prvoj
// polovini izvršavanja ptice hvataju plen krećući se ka drugim članovima jata i globalno najboljem (eksploracija),
// a u drugoj polovini beže od predatora skaliranim korakom u nasumičnom smeru (eksploatacija)
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func SecretaryBird(problem models.Problem) (result models.Result, err error) {

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

	population := secretaryBirdPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := secretaryBirdMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := secretaryBirdTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	birds := make([][]float64, population)
	values := make([]float64, population)
	for i := range birds {
		bird := make([]float64, dimensions)
		for d := range bird {
			bird[d] = secretaryBirdRangeLow + rand.Float64()*(secretaryBirdRangeHigh-secretaryBirdRangeLow)
		}
		birds[i] = bird
		values[i] = fn.Evaluate(bird)
	}

	best := append([]float64(nil), birds[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), birds[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range birds {
			if steps < maxSteps/2 {
				r1 := birds[rand.IntN(population)]
				r2 := birds[rand.IntN(population)]
				for d := range birds[i] {
					birds[i][d] = birds[i][d] + rand.Float64()*(r1[d]-birds[i][d]*rand.Float64()) + rand.Float64()*(best[d]-r2[d]*rand.Float64())
				}
			} else {
				R := 0.5 + rand.Float64()*0.5
				k := 1.0
				if rand.Float64() < 0.5 {
					k = -1
				}
				for d := range birds[i] {
					birds[i][d] = best[d] + R*k*birds[i][d]
				}
			}

			for d := range birds[i] {
				birds[i][d] = clamp(birds[i][d], secretaryBirdRangeLow, secretaryBirdRangeHigh)
			}

			values[i] = fn.Evaluate(birds[i])
			if values[i] < bestValue {
				bestValue = values[i]
				best = append([]float64(nil), birds[i]...)
			}
			if includeTrajectory {
				trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
			}
		}

		if math.Abs(bestValue-prevBestValue) < tolerance {
			noImprove++
		} else {
			noImprove = 0
		}
		if noImprove >= maxNoImprove {
			steps++
			break
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "secretary_bird",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
