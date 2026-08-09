package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	penguinSearchPopulation = 20
	penguinSearchMaxSteps   = 1000
	penguinSearchTolerance  = 1e-6
	penguinSearchRangeLow   = -5.0
	penguinSearchRangeHigh  = 5.0
)

// PenguinSearch minimizuje konfigurisanu benchmark funkciju koristeći Penguin Search Optimization Algorithm: svaki
// pingvin spiralno roni oko najboljeg pronađenog ribolovnog mesta, poluprečnikom rona koji se sužava tokom
// izvršavanja
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func PenguinSearch(problem models.Problem) (result models.Result, err error) {

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

	population := penguinSearchPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := penguinSearchMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := penguinSearchTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	penguins := make([][]float64, population)
	values := make([]float64, population)
	for i := range penguins {
		penguin := make([]float64, dimensions)
		for d := range penguin {
			penguin[d] = penguinSearchRangeLow + rand.Float64()*(penguinSearchRangeHigh-penguinSearchRangeLow)
		}
		penguins[i] = penguin
		values[i] = fn.Evaluate(penguin)
	}

	best := append([]float64(nil), penguins[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), penguins[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		// Originalna formula je u potpunosti zamenjivala poziciju svakim korakom (best + spiralni šum),
		// bez ikakvog inkrementalnog pomeraja — bezuslovna resempl-oko-best pretraga lako naiđe na duge
		// platoe bez poboljšanja pre nego što noImprove logika prekine izvršavanje. Zamenjeno je pravim
		// inkrementalnim pomerajem (deo puta ka best-u) uz opadajuću spiralnu perturbaciju
		frac := 1 - float64(steps)/float64(maxSteps)
		radius := 2 * frac * frac

		for i := range penguins {
			angle := rand.Float64() * 2 * math.Pi
			for d := range penguins[i] {
				penguins[i][d] = penguins[i][d] + 0.7*(best[d]-penguins[i][d]) + 0.5*radius*math.Cos(angle+float64(d))*rand.Float64()
				penguins[i][d] = clamp(penguins[i][d], penguinSearchRangeLow, penguinSearchRangeHigh)
			}
			values[i] = fn.Evaluate(penguins[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), penguins[i]...)
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
		Method:     "penguin_search",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
