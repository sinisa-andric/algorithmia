package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	sandCatPopulation = 20
	sandCatMaxSteps   = 1000
	sandCatTolerance  = 1e-6
	sandCatRangeLow   = -5.0
	sandCatRangeHigh  = 5.0
)

// SandCat minimizuje konfigurisanu benchmark funkciju koristeći Sand Cat Swarm Optimization: svaki mačak se kreće po
// kružnoj putanji oko najboljeg rešenja — kosinusnom komponentom u fazi eksploracije ili sinusnom u fazi
// eksploatacije — sa poluprečnikom koji linearno opada tokom izvršavanja
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func SandCat(problem models.Problem) (result models.Result, err error) {

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

	population := sandCatPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := sandCatMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := sandCatTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	cats := make([][]float64, population)
	values := make([]float64, population)
	for i := range cats {
		cat := make([]float64, dimensions)
		for d := range cat {
			cat[d] = sandCatRangeLow + rand.Float64()*(sandCatRangeHigh-sandCatRangeLow)
		}
		cats[i] = cat
		values[i] = fn.Evaluate(cat)
	}

	best := append([]float64(nil), cats[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), cats[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		r := 2 - float64(steps)*(2/float64(maxSteps))

		for i := range cats {
			radius := r * rand.Float64()
			thetaRad := rand.Float64() * 360 * math.Pi / 180

			if rand.Float64() <= 0.5 {
				for d := range cats[i] {
					cats[i][d] = best[d] + radius*math.Cos(thetaRad)*math.Abs(rand.Float64()*best[d]-cats[i][d])
				}
			} else {
				for d := range cats[i] {
					cats[i][d] = best[d] + radius*math.Sin(thetaRad)*math.Abs(rand.Float64()*best[d]-cats[i][d])
				}
			}

			for d := range cats[i] {
				cats[i][d] = clamp(cats[i][d], sandCatRangeLow, sandCatRangeHigh)
			}

			values[i] = fn.Evaluate(cats[i])
			if values[i] < bestValue {
				bestValue = values[i]
				best = append([]float64(nil), cats[i]...)
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
		Method:     "sand_cat",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
