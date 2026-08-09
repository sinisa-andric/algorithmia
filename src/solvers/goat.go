package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	goatPopulation = 20
	goatMaxSteps   = 1000
	goatTolerance  = 1e-6
	goatRangeLow   = -5.0
	goatRangeHigh  = 5.0
)

// Goat minimizuje konfigurisanu benchmark funkciju koristeći Goat Herd Optimization: koza se ili penje ka najboljoj
// kozi (eksploatacija) ili skače na susednu stenu istražujući u odnosu na nasumičnu kozu, uz malu dodatnu privlačnost
// ka najboljem rešenju da bi se sprečio slepi kolaps roja
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Goat(problem models.Problem) (result models.Result, err error) {

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

	population := goatPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := goatMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := goatTolerance
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

	goats := make([][]float64, population)
	values := make([]float64, population)
	for i := range goats {
		goat := make([]float64, dimensions)
		for d := range goat {
			goat[d] = goatRangeLow + rand.Float64()*(goatRangeHigh-goatRangeLow)
		}
		goats[i] = goat
		values[i] = fn.Evaluate(goat)
	}

	best := append([]float64(nil), goats[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), goats[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range goats {
			if rand.Float64() < 0.5 {
				for d := range goats[i] {
					goats[i][d] = goats[i][d] + rand.Float64()*(best[d]-goats[i][d])
				}
			} else {
				// čisto privlačenje ka nasumičnoj kozi je kontraktivna dinamika koja bi kolabirala roj u
				// proizvoljnu tačku — dodata je slaba dodatna privlačnost ka best-u da se to spreči
				r := goats[randomIndexExcept(i)]
				for d := range goats[i] {
					goats[i][d] = goats[i][d] + (rand.Float64()*2-1)*(r[d]-goats[i][d])*0.5 + rand.Float64()*(best[d]-goats[i][d])*0.1
				}
			}

			for d := range goats[i] {
				goats[i][d] = clamp(goats[i][d], goatRangeLow, goatRangeHigh)
			}
			values[i] = fn.Evaluate(goats[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), goats[i]...)
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
		Method:     "goat",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
