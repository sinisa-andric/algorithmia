package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	seaTurtlePopulation = 20
	seaTurtleMaxSteps   = 1000
	seaTurtleTolerance  = 1e-6
	seaTurtleRangeLow   = -5.0
	seaTurtleRangeHigh  = 5.0
)

// SeaTurtle minimizuje konfigurisanu benchmark funkciju koristeći Sea Turtle Foraging Algorithm: svaka kornjača se
// kreće ka najboljem rešenju uz dodatni zajednički pravac okeanske struje koji se nasumično bira svakog koraka
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func SeaTurtle(problem models.Problem) (result models.Result, err error) {

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

	population := seaTurtlePopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := seaTurtleMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := seaTurtleTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	turtles := make([][]float64, population)
	values := make([]float64, population)
	for i := range turtles {
		turtle := make([]float64, dimensions)
		for d := range turtle {
			turtle[d] = seaTurtleRangeLow + rand.Float64()*(seaTurtleRangeHigh-seaTurtleRangeLow)
		}
		turtles[i] = turtle
		values[i] = fn.Evaluate(turtle)
	}

	best := append([]float64(nil), turtles[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), turtles[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		current := make([]float64, dimensions)
		for d := range current {
			current[d] = rand.Float64()*2 - 1
		}

		for i := range turtles {
			for d := range turtles[i] {
				turtles[i][d] = turtles[i][d] + rand.Float64()*(best[d]-turtles[i][d]) + 0.1*current[d]
				turtles[i][d] = clamp(turtles[i][d], seaTurtleRangeLow, seaTurtleRangeHigh)
			}
			values[i] = fn.Evaluate(turtles[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), turtles[i]...)
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
		Method:     "sea_turtle",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
