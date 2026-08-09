package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	tunicatePopulation = 20
	tunicateMaxSteps   = 1000
	tunicateTolerance  = 1e-6
	tunicateRangeLow   = -5.0
	tunicateRangeHigh  = 5.0
)

// Tunicate minimizuje konfigurisanu benchmark funkciju koristeći Tunicate Swarm Algorithm: tunikat se ili mlazno
// pokreće ka najboljem rešenju ili se ponaša rojno prateći nasumičnog suseda,
// u zavisnosti od nasumičnog faktora struje
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Tunicate(problem models.Problem) (result models.Result, err error) {

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

	population := tunicatePopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := tunicateMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := tunicateTolerance
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

	tunicates := make([][]float64, population)
	values := make([]float64, population)
	for i := range tunicates {
		tunicate := make([]float64, dimensions)
		for d := range tunicate {
			tunicate[d] = tunicateRangeLow + rand.Float64()*(tunicateRangeHigh-tunicateRangeLow)
		}
		tunicates[i] = tunicate
		values[i] = fn.Evaluate(tunicate)
	}

	best := append([]float64(nil), tunicates[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), tunicates[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range tunicates {
			c1 := rand.Float64()
			c2 := rand.Float64()
			c3 := rand.Float64()

			if c3 >= 0.5 {
				for d := range tunicates[i] {
					dist := math.Abs(c2*best[d] - tunicates[i][d])
					tunicates[i][d] = best[d] - c1*dist
				}
			} else {
				// Čisto privlačenje ka nasumičnom peer-u (bez ikakve veze sa best-om) je
				// kontraktivna dinamika — udaljenost r[d]-tunicates[i][d] se eksponencijalno
				// smanjuje iz koraka u korak, pa roj brzo kolabira (gubi diverzitet) u proizvoljnu
				// tačku daleko od pravog minimuma i tu ostaje trajno zaglavljen. Ravnopravna
				// privlačnost ka best-u drži roj u toku sa najboljim pronađenim rešenjem (isti fix
				// kao ranije za chameleon.go).
				r := tunicates[randomIndexExcept(i)]
				for d := range tunicates[i] {
					tunicates[i][d] = tunicates[i][d] + rand.Float64()*(r[d]-tunicates[i][d]) + rand.Float64()*(best[d]-tunicates[i][d])
				}
			}

			for d := range tunicates[i] {
				tunicates[i][d] = clamp(tunicates[i][d], tunicateRangeLow, tunicateRangeHigh)
			}
			values[i] = fn.Evaluate(tunicates[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), tunicates[i]...)
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
		Method:     "tunicate",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
