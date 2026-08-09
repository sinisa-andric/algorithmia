package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	dolphinPopulation = 20
	dolphinMaxSteps   = 1000
	dolphinTolerance  = 1e-6
	dolphinRangeLow   = -5.0
	dolphinRangeHigh  = 5.0
)

// Dolphin minimizuje konfigurisanu benchmark funkciju koristeći Dolphin Echolocation Optimization: svaki delfin se
// pozicionira oko najboljeg rešenja unutar efektivnog radijusa eholokacije koji se sužava tokom izvršavanja, uz
// malu dodatnu komponentu koja čuva deo prethodne pozicije
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Dolphin(problem models.Problem) (result models.Result, err error) {

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

	population := dolphinPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := dolphinMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := dolphinTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	dolphins := make([][]float64, population)
	values := make([]float64, population)
	for i := range dolphins {
		dolphin := make([]float64, dimensions)
		for d := range dolphin {
			dolphin[d] = dolphinRangeLow + rand.Float64()*(dolphinRangeHigh-dolphinRangeLow)
		}
		dolphins[i] = dolphin
		values[i] = fn.Evaluate(dolphin)
	}

	best := append([]float64(nil), dolphins[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), dolphins[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		// Originalna formula je u potpunosti ZAMENJIVALA poziciju svakim korakom (best + šum*radius),
		// bez ikakvog inkrementalnog pomeraja — to je čista bezuslovna resempl-oko-best pretraga koja
		// lako naiđe na duge platoe bez poboljšanja (50+ koraka), pa noImprove logika prekine izvršavanje
		// davno pre stizanja blizu optimuma. Zamenjeno je pravim inkrementalnim pomerajem (deo puta ka
		// best-u svaki korak) uz opadajuću nasumičnu perturbaciju koja predstavlja radijus eholokacije
		frac := 1 - float64(steps)/float64(maxSteps)
		radius := (dolphinRangeHigh - dolphinRangeLow) * 0.5 * frac * frac

		for i := range dolphins {
			for d := range dolphins[i] {
				dolphins[i][d] = dolphins[i][d] + 0.7*(best[d]-dolphins[i][d]) + (rand.Float64()*2-1)*radius*0.1
				dolphins[i][d] = clamp(dolphins[i][d], dolphinRangeLow, dolphinRangeHigh)
			}
			values[i] = fn.Evaluate(dolphins[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), dolphins[i]...)
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
		Method:     "dolphin",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
