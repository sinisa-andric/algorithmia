package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
)

const (
	atomSearchPopulation    = 20
	atomSearchMaxSteps      = 500
	atomSearchAlpha         = 50.0
	atomSearchBeta          = 0.2
	atomSearchTolerance     = 1e-6
	atomSearchRangeLow      = -5.0
	atomSearchRangeHigh     = 5.0
	atomSearchForceBound    = 1.0
	atomSearchVelocityBound = 1.0
)

// AtomSearch minimizuje konfigurisanu benchmark funkciju koristeći Atom Search Optimization: atomi se privlače i
// odbijaju od nekoliko najbližih suseda preko Lenard-Džonsovog potencijala, uz dodatnu geometrijsku vezu ka
// najlakšem (najboljem) atomu koja deluje kao ograničenje veze
// problem.Point inicijalizuje populaciju atoma i određuje njenu dimenzionalnost
func AtomSearch(problem models.Problem) (result models.Result, err error) {

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

	population := atomSearchPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := atomSearchMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	alpha := atomSearchAlpha
	if v, ok := problem.Payload["alpha"].(float64); ok {
		alpha = v
	}

	beta := atomSearchBeta
	if v, ok := problem.Payload["beta"].(float64); ok {
		beta = v
	}

	tolerance := atomSearchTolerance
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
			individual[d] = atomSearchRangeLow + rand.Float64()*(atomSearchRangeHigh-atomSearchRangeLow)
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

		worstValue := values[0]
		for _, v := range values {
			if v > worstValue {
				worstValue = v
			}
		}

		mass := make([]float64, population)
		totalMass := 0.0
		for i, v := range values {
			mass[i] = math.Exp(-(v - bestValue) / (worstValue - bestValue + 1e-10))
			totalMass += mass[i]
		}
		meanMass := totalMass / float64(population)
		for i := range mass {
			mass[i] /= meanMass
		}

		k := population - int(float64(steps)/float64(maxSteps)*float64(population-2))
		if k < 2 {
			k = 2
		}
		if k > population {
			k = population
		}

		for i := range individuals {
			order := make([]int, population)
			for j := range order {
				order[j] = j
			}
			sort.Slice(order, func(a, b int) bool {
				diffA := make([]float64, dimensions)
				diffB := make([]float64, dimensions)
				for d := range diffA {
					diffA[d] = individuals[order[a]][d] - individuals[i][d]
					diffB[d] = individuals[order[b]][d] - individuals[i][d]
				}
				return norm(diffA) < norm(diffB)
			})

			ljForce := make([]float64, dimensions)
			for n := 0; n < k && n < population; n++ {
				j := order[n]
				if j == i {
					continue
				}
				diff := make([]float64, dimensions)
				for d := range diff {
					diff[d] = individuals[j][d] - individuals[i][d]
				}
				// LJ potencijal menja predznak (privlačenje <-> odbijanje) na r = (1/2)^(1/6) =~ 0.891 — sa
				// r normalizovanim samo širinom opsega, ta ravnotežna udaljenost odgovara skoro 90% širine
				// opsega, pa je populacija sistemski gurana da se raširi po skoro celom prostoru umesto da
				// konvergira, boreći se protiv constraint_force veze ka best-u (izmereno 5-11/20 zaglavljenih
				// na sphere). Deljenje širim faktorom drži r malim za sva realna rastojanja unutar opsega, tako
				// da je LJ sila skoro uvek privlačna i pojačava, umesto da se suprotstavlja, konvergenciji ka best-u
				r := norm(diff)/((atomSearchRangeHigh-atomSearchRangeLow)*5) + 1e-10
				lj := alpha * (1 - float64(steps)/float64(maxSteps)) * (2*math.Pow(r, 13) - math.Pow(r, 7))
				for d := range ljForce {
					ljForce[d] += lj * diff[d]
				}
			}

			for d := range ljForce {
				ljForce[d] = clamp(ljForce[d], -atomSearchForceBound, atomSearchForceBound)
			}

			constraintForce := make([]float64, dimensions)
			for d := range constraintForce {
				constraintForce[d] = beta * rand.Float64() * (best[d] - individuals[i][d])
			}

			for d := range individuals[i] {
				// masa normalizovana da suma preko populacije bude 1 (umesto proseka 1) je za population=20
				// davala deljenje silom ~20x prejakom, gurajući brzinu daleko iznad širine opsega (izmereno do
				// ~95 naspram opsega širine 10) i zaglavljujući sve jedinke na granici svaki korak — normalizacija
				// je promenjena da prosečna masa bude 1, uz eksplicitan bound na brzinu (kao gravitational_search.go)
				// kao dodatnu zaštitu. I dalje, čisto silom-vođeno kretanje bez ikakvog nezavisnog šuma je populaciju
				// dovodilo do LJ ravnotežnog klastera koji nije tačno na best-u i tu ostajalo zaglavljeno (izmereno
				// 5-9/20 zaglavljenih na sphere) — dodat je mali nezavisan šum da populacija može da pobegne iz
				// takve ravnoteže
				velocity[i][d] = rand.Float64()*velocity[i][d] + (ljForce[d]+constraintForce[d])/(mass[i]+1e-10)
				velocity[i][d] = clamp(velocity[i][d], -atomSearchVelocityBound, atomSearchVelocityBound)
				individuals[i][d] = clamp(individuals[i][d]+velocity[i][d]+(rand.Float64()*2-1)*0.05, atomSearchRangeLow, atomSearchRangeHigh)
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
		Method:     "atom_search",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
