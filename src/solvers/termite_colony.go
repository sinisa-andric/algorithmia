package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	termiteColonyPopulation = 20
	termiteColonyMaxSteps   = 1000
	termiteColonyTolerance  = 1e-6
	termiteColonyRangeLow   = -5.0
	termiteColonyRangeHigh  = 5.0
)

// TermiteColony minimizuje konfigurisanu benchmark funkciju koristeći Termite Colony Optimization: svaka termita se
// kreće ka drugoj termiti izabranoj ruletom proporcionalno feromonskom tragu,
// koji se ažurira obrnuto proporcionalno vrednosti funkcije nakon svakog pomeraja
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func TermiteColony(problem models.Problem) (result models.Result, err error) {

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

	population := termiteColonyPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := termiteColonyMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := termiteColonyTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	termites := make([][]float64, population)
	values := make([]float64, population)
	pheromone := make([]float64, population)
	for i := range termites {
		termite := make([]float64, dimensions)
		for d := range termite {
			termite[d] = termiteColonyRangeLow + rand.Float64()*(termiteColonyRangeHigh-termiteColonyRangeLow)
		}
		termites[i] = termite
		values[i] = fn.Evaluate(termite)
		pheromone[i] = 1.0
	}

	best := append([]float64(nil), termites[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), termites[i]...)
		}
	}

	selectByPheromone := func(exclude int) int {
		total := 0.0
		for i, p := range pheromone {
			if i == exclude {
				continue
			}
			total += p
		}
		if total <= 0 {
			j := rand.IntN(population)
			for j == exclude {
				j = rand.IntN(population)
			}
			return j
		}
		pick := rand.Float64() * total
		cumulative := 0.0
		for i, p := range pheromone {
			if i == exclude {
				continue
			}
			cumulative += p
			if cumulative >= pick {
				return i
			}
		}
		return exclude
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range termites {
			j := selectByPheromone(i)
			for d := range termites[i] {
				termites[i][d] = termites[i][d] + rand.Float64()*(termites[j][d]-termites[i][d])
				termites[i][d] = clamp(termites[i][d], termiteColonyRangeLow, termiteColonyRangeHigh)
			}
			values[i] = fn.Evaluate(termites[i])
			pheromone[i] = 1 / (values[i] + 1e-10)
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), termites[i]...)
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
		Method:     "termite_colony",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
