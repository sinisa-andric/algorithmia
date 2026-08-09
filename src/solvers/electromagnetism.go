package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	electromagnetismPopulation = 20
	electromagnetismMaxSteps   = 500
	electromagnetismTolerance  = 1e-6
	electromagnetismRangeLow   = -5.0
	electromagnetismRangeHigh  = 5.0
	electromagnetismSample     = 5
)

// Electromagnetism minimizuje konfigurisanu benchmark funkciju koristeći Electromagnetism-like Algorithm: svaka
// čestica se privlači ka nekoliko nasumično uzorkovanih boljih čestica i odbija od gorih, sa nabojem
// proporcionalnim kvalitetu čestice
// problem.Point inicijalizuje populaciju naelektrisanih čestica i određuje njenu dimenzionalnost
func Electromagnetism(problem models.Problem) (result models.Result, err error) {

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

	population := electromagnetismPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := electromagnetismMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := electromagnetismTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	individuals := make([][]float64, population)
	values := make([]float64, population)
	for i := range individuals {
		individual := make([]float64, dimensions)
		for d := range individual {
			individual[d] = electromagnetismRangeLow + rand.Float64()*(electromagnetismRangeHigh-electromagnetismRangeLow)
		}
		individuals[i] = individual
		values[i] = fn.Evaluate(individual)
	}

	best := append([]float64(nil), individuals[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), individuals[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		worstValue := values[0]
		for _, v := range values {
			if v > worstValue {
				worstValue = v
			}
		}

		charge := make([]float64, population)
		for i, v := range values {
			charge[i] = math.Exp(-(v - bestValue) / (worstValue - bestValue + 1e-10))
		}

		for i := range individuals {
			totalForce := make([]float64, dimensions)
			for range electromagnetismSample {
				j := rand.IntN(population)
				if j == i {
					continue
				}
				for d := range totalForce {
					diff := individuals[j][d] - individuals[i][d]
					if values[j] < values[i] {
						totalForce[d] += charge[j] * diff
					} else {
						totalForce[d] -= charge[j] * diff
					}
				}
			}

			for d := range individuals[i] {
				// za trenutno najbolju jedinku SVI uzorkovani peer-ovi su lošiji, pa je total_force uvek čisto
				// odbojan bez ikakve privlačne komponente — to sistemski gura upravo najbolju jedinku dalje od
				// optimuma bez ičega što bi je vratilo (izmereno 1-2/20 zaglavljenih na sphere). Mali eksplicitan
				// pull ka globalnom best-u daje svakoj jedinki vezu koja ne zavisi od ishoda uzorkovanja peer-ova
				pullBest := rand.Float64() * (best[d] - individuals[i][d]) * 0.05
				noise := (rand.Float64()*2 - 1) * 0.05
				individuals[i][d] = clamp(individuals[i][d]+rand.Float64()*totalForce[d]*0.1+pullBest+noise, electromagnetismRangeLow, electromagnetismRangeHigh)
			}
			values[i] = fn.Evaluate(individuals[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), individuals[i]...)
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
		Method:     "electromagnetism",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
