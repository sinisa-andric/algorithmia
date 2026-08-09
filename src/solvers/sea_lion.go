package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	seaLionPopulation = 20
	seaLionMaxSteps   = 1000
	seaLionTolerance  = 1e-6
	seaLionRangeLow   = -5.0
	seaLionRangeHigh  = 5.0
)

// SeaLion minimizuje konfigurisanu benchmark funkciju koristeći Sea Lion Optimization: morski lav se ili spiralno
// kreće oko najboljeg rešenja ili prati ažuriranje pokreta, oponašajući lov na plen praćenjem vodećeg člana čopora
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func SeaLion(problem models.Problem) (result models.Result, err error) {

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

	population := seaLionPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := seaLionMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := seaLionTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	seaLions := make([][]float64, population)
	values := make([]float64, population)
	for i := range seaLions {
		seaLion := make([]float64, dimensions)
		for d := range seaLion {
			seaLion[d] = seaLionRangeLow + rand.Float64()*(seaLionRangeHigh-seaLionRangeLow)
		}
		seaLions[i] = seaLion
		values[i] = fn.Evaluate(seaLion)
	}

	best := append([]float64(nil), seaLions[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), seaLions[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		mu := 0.5 + rand.Float64()*0.5

		for i := range seaLions {
			if rand.Float64() < 0.5 {
				for d := range seaLions[i] {
					dist := math.Abs(best[d] - seaLions[i][d])
					seaLions[i][d] = dist*math.Cos(2*math.Pi*rand.Float64()) + best[d]
				}
			} else {
				for d := range seaLions[i] {
					seaLions[i][d] = seaLions[i][d] + mu*(best[d]-seaLions[i][d])
				}
			}

			for d := range seaLions[i] {
				seaLions[i][d] = clamp(seaLions[i][d], seaLionRangeLow, seaLionRangeHigh)
			}
			values[i] = fn.Evaluate(seaLions[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), seaLions[i]...)
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
		Method:     "sea_lion",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
