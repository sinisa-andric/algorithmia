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
	paddyFieldPopulation     = 20
	paddyFieldMaxSteps       = 500
	paddyFieldSelectionRatio = 0.3
	paddyFieldTolerance      = 1e-6
	paddyFieldRangeLow       = -5.0
	paddyFieldRangeHigh      = 5.0
)

// PaddyField minimizuje konfigurisanu benchmark funkciju koristeći Paddy Field Algorithm: najbolje rangirana
// semena svakog koraka proizvode broj novih semena proporcionalan sopstvenom rangu i rasipaju se u opadajućem
// radijusu oko roditelja, dok se najgora semena u populaciji zamenjuju novoproizvedenim
// problem.Point inicijalizuje populaciju semena i određuje njenu dimenzionalnost
func PaddyField(problem models.Problem) (result models.Result, err error) {

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

	population := paddyFieldPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := paddyFieldMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	selectionRatio := paddyFieldSelectionRatio
	if v, ok := problem.Payload["selection_ratio"].(float64); ok {
		selectionRatio = v
	}

	tolerance := paddyFieldTolerance
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
			individual[d] = paddyFieldRangeLow + rand.Float64()*(paddyFieldRangeHigh-paddyFieldRangeLow)
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

		order := make([]int, len(individuals))
		for i := range order {
			order[i] = i
		}
		sort.Slice(order, func(a, b int) bool {
			return values[order[a]] < values[order[b]]
		})

		granica := int(math.Round(selectionRatio * float64(len(individuals))))
		if granica < 1 {
			granica = 1
		}
		if granica > len(individuals) {
			granica = len(individuals)
		}

		newSeeds := make([][]float64, 0, granica*granica)
		newValues := make([]float64, 0, granica*granica)
		for rank := 0; rank < granica; rank++ {
			idx := order[rank]
			numNew := granica - rank
			for s := 0; s < numNew; s++ {
				seed := make([]float64, dimensions)
				for d := range seed {
					amplitude := 0.5 * (1 - float64(steps)/float64(maxSteps))
					seed[d] = clamp(individuals[idx][d]+(rand.Float64()*2-1)*amplitude, paddyFieldRangeLow, paddyFieldRangeHigh)
				}
				newSeeds = append(newSeeds, seed)
				newValues = append(newValues, fn.Evaluate(seed))
			}
		}

		// zameni najgora semena novoproizvedenim, uz zadržavanje veličine population bez obzira na to koliko je
		// semena proizvedeno ovog koraka — broj novih semena varira sa granicom i rangom, pa se ovde ne oslanjamo
		// na fiksne indekse već na dužinu isečaka i dopunu/odsecanje do tačne veličine
		numKeep := population - len(newSeeds)
		if numKeep < 0 {
			numKeep = 0
		}
		if numKeep > len(order) {
			numKeep = len(order)
		}

		nextIndividuals := make([][]float64, 0, population)
		nextValues := make([]float64, 0, population)
		for r := 0; r < numKeep; r++ {
			idx := order[r]
			nextIndividuals = append(nextIndividuals, individuals[idx])
			nextValues = append(nextValues, values[idx])
		}
		nextIndividuals = append(nextIndividuals, newSeeds...)
		nextValues = append(nextValues, newValues...)

		if len(nextIndividuals) > population {
			nextIndividuals = nextIndividuals[:population]
			nextValues = nextValues[:population]
		}
		for len(nextIndividuals) < population {
			individual := make([]float64, dimensions)
			for d := range individual {
				individual[d] = paddyFieldRangeLow + rand.Float64()*(paddyFieldRangeHigh-paddyFieldRangeLow)
			}
			nextIndividuals = append(nextIndividuals, individual)
			nextValues = append(nextValues, fn.Evaluate(individual))
		}

		individuals = nextIndividuals
		values = nextValues

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), individuals[i]...)
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
		Method:     "paddy_field",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
