package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	fruitFlyPopulation = 20
	fruitFlyMaxSteps   = 1000
	fruitFlyTolerance  = 1e-6
	fruitFlyRangeLow   = -5.0
	fruitFlyRangeHigh  = 5.0
)

// FruitFly minimizuje konfigurisanu benchmark funkciju koristeći Fruit Fly Optimization Algorithm: oko najbolje
// pronađene pozicije se nasumično pretražuje miris (osmatranje putem koncentracije mirisa),
// a zatim se svaka muva vizuelno kreće ka trenutno najboljem rešenju
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func FruitFly(problem models.Problem) (result models.Result, err error) {

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

	population := fruitFlyPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := fruitFlyMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := fruitFlyTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	fruitFlies := make([][]float64, population)
	values := make([]float64, population)
	for i := range fruitFlies {
		fly := make([]float64, dimensions)
		for d := range fly {
			fly[d] = fruitFlyRangeLow + rand.Float64()*(fruitFlyRangeHigh-fruitFlyRangeLow)
		}
		fruitFlies[i] = fly
		values[i] = fn.Evaluate(fly)
	}

	best := append([]float64(nil), fruitFlies[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), fruitFlies[i]...)
		}
	}
	bestPos := append([]float64(nil), best...)

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range fruitFlies {
			novi := make([]float64, dimensions)
			for d := range novi {
				// osmatranje mirisa: X/Y su nasumični kandidati oko best_pos, S (koncentracija
				// mirisa) je obrnuto proporcionalna udaljenosti i skalira konačni vizuelni skok
				x := bestPos[d] + rand.Float64()
				y := bestPos[d] + rand.Float64()
				dist := math.Sqrt(x*x + y*y)
				s := 1 / (dist + 1e-10)

				novi[d] = clamp(bestPos[d]+(rand.Float64()*2-1)*2.0*s, fruitFlyRangeLow, fruitFlyRangeHigh)
			}
			noviValue := fn.Evaluate(novi)
			if noviValue < bestValue {
				bestValue = noviValue
				best = novi
				bestPos = append([]float64(nil), novi...)
			}

			for d := range fruitFlies[i] {
				fruitFlies[i][d] = clamp(best[d]+rand.Float64()*(best[d]-fruitFlies[i][d]), fruitFlyRangeLow, fruitFlyRangeHigh)
			}
			values[i] = fn.Evaluate(fruitFlies[i])
			if values[i] < bestValue {
				bestValue = values[i]
				best = append([]float64(nil), fruitFlies[i]...)
				bestPos = append([]float64(nil), best...)
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
		Method:     "fruit_fly",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
