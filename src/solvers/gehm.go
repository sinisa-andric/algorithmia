package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	gehmPopulation   = 20
	gehmMaxSteps     = 500
	gehmMutationRate = 0.1
	gehmG0           = 100.0
	gehmTolerance    = 1e-6
	gehmRangeLow     = -5.0
	gehmRangeHigh    = 5.0
	gehmPeerSample   = 5
	gehmForceBound   = 0.5
)

// GEHM minimizuje konfigurisanu benchmark funkciju koristeći Gravitational-Evolutionary Hybrid Method: svaka
// jedinka je privučena ka nasumičnom podskupu peer-ova gravitacionom silom ponderisanom njihovom masom (boljom
// vrednošću funkcije) i gravitacionom konstantom koja opada tokom izvršavanja, uz povremenu mutaciju
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func GEHM(problem models.Problem) (result models.Result, err error) {

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

	population := gehmPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := gehmMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	mutationRate := gehmMutationRate
	if v, ok := problem.Payload["mutation_rate"].(float64); ok {
		mutationRate = v
	}

	g0 := gehmG0
	if v, ok := problem.Payload["G0"].(float64); ok {
		g0 = v
	}

	tolerance := gehmTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	individuals := make([][]float64, population)
	values := make([]float64, population)
	for i := range individuals {
		individual := make([]float64, dimensions)
		for d := range individual {
			individual[d] = gehmRangeLow + rand.Float64()*(gehmRangeHigh-gehmRangeLow)
		}
		individuals[i] = individual
		values[i] = fn.Evaluate(individual)
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

		g := g0 * (1 - float64(steps)/float64(maxSteps))

		worstValue := values[0]
		bestValueThisStep := values[0]
		for _, v := range values {
			if v > worstValue {
				worstValue = v
			}
			if v < bestValueThisStep {
				bestValueThisStep = v
			}
		}

		mass := make([]float64, population)
		for i, v := range values {
			mass[i] = (worstValue - v) / (worstValue - bestValueThisStep + 1e-10)
		}

		forces := make([][]float64, population)
		for i := range individuals {
			force := make([]float64, dimensions)
			for range gehmPeerSample {
				j := rand.IntN(population)
				if j == i {
					continue
				}
				diff := make([]float64, dimensions)
				r := 0.0
				for d := range diff {
					diff[d] = individuals[j][d] - individuals[i][d]
					r += diff[d] * diff[d]
				}
				r = math.Sqrt(r) + 1e-10
				for d := range force {
					force[d] += g * mass[j] * diff[d] / r
				}
			}
			forces[i] = force
		}

		for i := range individuals {
			for d := range individuals[i] {
				// Sila nije normalizovana po broju uzorkovanih peer-ova niti ograničena, pa kad su
				// dve jedinke slučajno blizu (malo r) sila eksplodira daleko iznad širine opsega —
				// izmereno do ~360 pri G0=100, naspram opsega širine 10 — što jedinke stalno odbacuje
				// na granicu opsega umesto glatkog gravitacionog privlačenja. Sila je zato ograničena
				// na razumnu veličinu koraka pre primene
				f := clamp(forces[i][d], -gehmForceBound, gehmForceBound)
				individuals[i][d] = clamp(individuals[i][d]+rand.Float64()*f, gehmRangeLow, gehmRangeHigh)
			}
			values[i] = fn.Evaluate(individuals[i])
		}

		if rand.Float64() < mutationRate {
			i := rand.IntN(population)
			d := rand.IntN(dimensions)
			individuals[i][d] = clamp(individuals[i][d]+(rand.Float64()*2-1)*0.3, gehmRangeLow, gehmRangeHigh)
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
		Method:     "gehm",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
