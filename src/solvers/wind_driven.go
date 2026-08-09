package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	windDrivenPopulation  = 20
	windDrivenMaxSteps    = 500
	windDrivenRT          = 3.0
	windDrivenGConst      = 0.2
	windDrivenAlphaC      = 0.4
	windDrivenTolerance   = 1e-6
	windDrivenRangeLow    = -5.0
	windDrivenRangeHigh   = 5.0
	windDrivenVelocityMax = 0.3
)

// WindDriven minimizuje konfigurisanu benchmark funkciju koristeći Wind Driven Optimization: vazdušni paketi nose
// brzinu oblikovanu trenjem, gravitacijom ka centru, pritisak-gradijentom (Koriolisova sila) ka best-u i vezom ka
// nasumičnom peer-u
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func WindDriven(problem models.Problem) (result models.Result, err error) {

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

	population := windDrivenPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := windDrivenMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	rt := windDrivenRT
	if v, ok := problem.Payload["RT"].(float64); ok {
		rt = v
	}

	gConst := windDrivenGConst
	if v, ok := problem.Payload["g_const"].(float64); ok {
		gConst = v
	}

	alphaC := windDrivenAlphaC
	if v, ok := problem.Payload["alpha_c"].(float64); ok {
		alphaC = v
	}

	tolerance := windDrivenTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	individuals := make([][]float64, population)
	values := make([]float64, population)
	velocity := make([][]float64, population)
	for i := range individuals {
		individual := make([]float64, dimensions)
		for d := range individual {
			individual[d] = windDrivenRangeLow + rand.Float64()*(windDrivenRangeHigh-windDrivenRangeLow)
		}
		individuals[i] = individual
		values[i] = fn.Evaluate(individual)
		velocity[i] = make([]float64, dimensions)
	}

	best := append([]float64(nil), individuals[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), individuals[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range individuals {
			peerIdx := i
			if population > 1 {
				peerIdx = rand.IntN(population)
				for peerIdx == i {
					peerIdx = rand.IntN(population)
				}
			}

			for d := range individuals[i] {
				// rtTerm = -RT*velocity uz inerciju 0.5 daje efektivni koeficijent (0.5-RT) u rekurziji
				// brzine — za podrazumevani RT=3.0 to je -2.5, po apsolutnoj vrednosti > 1, pa je rekurzija
				// nestabilna i brzina osciluje između granica bez obzira na eksplicitan bound (izmereno: sve
				// jedinke zaglavljene oko reziduala ~0.04-0.29 umesto konvergencije ka 0 na sphere, potvrđeno i
				// preko /benchmark analize kao "zaglavljen" u 14/15 pokušaja). Trenje je skalirano tako da
				// koeficijent ostane stabilan (< 1 po apsolutnoj vrednosti) za bilo koju razumnu vrednost RT
				rtTerm := -0.2 * rt * velocity[i][d]
				gravityTerm := -gConst * individuals[i][d]
				coriolisTerm := rt * (best[d] - individuals[i][d])
				otherTerm := alphaC * rand.Float64() * (individuals[peerIdx][d] - individuals[i][d])

				velocity[i][d] = 0.5*velocity[i][d] + rtTerm + gravityTerm + coriolisTerm + otherTerm
				velocity[i][d] = clamp(velocity[i][d], -windDrivenVelocityMax, windDrivenVelocityMax)

				individuals[i][d] = clamp(individuals[i][d]+velocity[i][d], windDrivenRangeLow, windDrivenRangeHigh)
			}
			values[i] = fn.Evaluate(individuals[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), individuals[i]...)
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
		Method:     "wind_driven",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
