package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	starfishPopulation = 20
	starfishMaxSteps   = 1000
	starfishTolerance  = 1e-6
	starfishRangeLow   = -5.0
	starfishRangeHigh  = 5.0
)

// Starfish minimizuje konfigurisanu benchmark funkciju koristeći Starfish Optimization Algorithm: morska zvezda se
// ili bidirekciono istražuje duž nasumično izabrane ose amplitudom koja opada tokom izvršavanja, ili se regeneriše
// krećući se ka najboljem rešenju
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Starfish(problem models.Problem) (result models.Result, err error) {

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

	population := starfishPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := starfishMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := starfishTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	starfish := make([][]float64, population)
	values := make([]float64, population)
	for i := range starfish {
		s := make([]float64, dimensions)
		for d := range s {
			s[d] = starfishRangeLow + rand.Float64()*(starfishRangeHigh-starfishRangeLow)
		}
		starfish[i] = s
		values[i] = fn.Evaluate(s)
	}

	best := append([]float64(nil), starfish[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), starfish[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range starfish {
			if rand.Float64() < 0.5 {
				d := rand.IntN(dimensions)
				starfish[i][d] = starfish[i][d] + (rand.Float64()*2-1)*(1-float64(steps)/float64(maxSteps))*2
			} else {
				for d := range starfish[i] {
					starfish[i][d] = starfish[i][d] + rand.Float64()*(best[d]-starfish[i][d])
				}
			}

			for d := range starfish[i] {
				starfish[i][d] = clamp(starfish[i][d], starfishRangeLow, starfishRangeHigh)
			}
			values[i] = fn.Evaluate(starfish[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), starfish[i]...)
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
		Method:     "starfish",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
