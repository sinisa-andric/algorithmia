package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	coralReefPopulation = 20
	coralReefMaxSteps   = 1000
	coralReefTolerance  = 1e-6
	coralReefRangeLow   = -5.0
	coralReefRangeHigh  = 5.0
)

// CoralReef minimizuje konfigurisanu benchmark funkciju koristeći Coral Reef Optimization: koral se ili
// rekombinuje sa nasumičnim peer-om po dimenzijama (broadcast spawning) ili se aseksualno pomera ka najboljem
// rešenju (budding), a najgori koral u populaciji se povremeno nasumično resetuje (depredation)
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func CoralReef(problem models.Problem) (result models.Result, err error) {

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

	population := coralReefPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := coralReefMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := coralReefTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	randomIndexExcept := func(exclude int) int {
		if population < 2 {
			return exclude
		}
		j := rand.IntN(population)
		for j == exclude {
			j = rand.IntN(population)
		}
		return j
	}

	corals := make([][]float64, population)
	values := make([]float64, population)
	for i := range corals {
		coral := make([]float64, dimensions)
		for d := range coral {
			coral[d] = coralReefRangeLow + rand.Float64()*(coralReefRangeHigh-coralReefRangeLow)
		}
		corals[i] = coral
		values[i] = fn.Evaluate(coral)
	}

	best := append([]float64(nil), corals[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), corals[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range corals {
			if rand.Float64() < 0.5 {
				r := corals[randomIndexExcept(i)]
				for d := range corals[i] {
					if rand.Float64() < 0.5 {
						corals[i][d] = r[d]
					}
				}
			} else {
				for d := range corals[i] {
					corals[i][d] = corals[i][d] + rand.Float64()*(best[d]-corals[i][d])
				}
			}

			for d := range corals[i] {
				corals[i][d] = clamp(corals[i][d], coralReefRangeLow, coralReefRangeHigh)
			}
			values[i] = fn.Evaluate(corals[i])
		}

		worstIdx := 0
		for i, v := range values {
			if v > values[worstIdx] {
				worstIdx = i
			}
		}
		for d := range corals[worstIdx] {
			corals[worstIdx][d] = coralReefRangeLow + rand.Float64()*(coralReefRangeHigh-coralReefRangeLow)
		}
		values[worstIdx] = fn.Evaluate(corals[worstIdx])

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), corals[i]...)
			}
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
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
		Method:     "coral_reef",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
