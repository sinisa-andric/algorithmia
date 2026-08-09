package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	spiderMonkeyPopulation = 20
	spiderMonkeyMaxSteps   = 1000
	spiderMonkeyTolerance  = 1e-6
	spiderMonkeyRangeLow   = -5.0
	spiderMonkeyRangeHigh  = 5.0
	spiderMonkeyNumGroups  = 4
)

// SpiderMonkey minimizuje konfigurisanu benchmark funkciju koristeći Spider Monkey Optimization: populacija se deli
// na grupe (fission-fusion struktura), svaki majmun se u Local Leader fazi kreće ka vođi svoje grupe i nasumičnom
// članu grupe, a u Global Leader fazi (sa verovatnoćom proporcionalnom fitnesu) ka globalno najboljem rešenju
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func SpiderMonkey(problem models.Problem) (result models.Result, err error) {

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

	population := spiderMonkeyPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := spiderMonkeyMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := spiderMonkeyTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	numGroups := spiderMonkeyNumGroups
	if v, ok := problem.Payload["num_groups"].(float64); ok {
		numGroups = int(v)
	}
	numGroups = max(1, min(numGroups, population))

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	monkeys := make([][]float64, population)
	values := make([]float64, population)
	for i := range monkeys {
		monkey := make([]float64, dimensions)
		for d := range monkey {
			monkey[d] = spiderMonkeyRangeLow + rand.Float64()*(spiderMonkeyRangeHigh-spiderMonkeyRangeLow)
		}
		monkeys[i] = monkey
		values[i] = fn.Evaluate(monkey)
	}

	groups := make([][]int, numGroups)
	for i := range monkeys {
		g := i % numGroups
		groups[g] = append(groups[g], i)
	}

	localLeader := make([][]float64, numGroups)
	localLeaderValue := make([]float64, numGroups)
	updateLocalLeader := func(g int) {
		bestIdx := groups[g][0]
		bestVal := values[bestIdx]
		for _, idx := range groups[g] {
			if values[idx] < bestVal {
				bestVal = values[idx]
				bestIdx = idx
			}
		}
		localLeader[g] = append([]float64(nil), monkeys[bestIdx]...)
		localLeaderValue[g] = bestVal
	}
	for g := range groups {
		updateLocalLeader(g)
	}

	globalLeader := append([]float64(nil), monkeys[0]...)
	globalLeaderValue := values[0]
	updateGlobalLeader := func() {
		for i, v := range values {
			if v < globalLeaderValue {
				globalLeaderValue = v
				globalLeader = append([]float64(nil), monkeys[i]...)
			}
		}
	}
	updateGlobalLeader()

	randomFromGroupExcept := func(g, exclude int) []float64 {
		members := groups[g]
		if len(members) < 2 {
			return monkeys[exclude]
		}
		j := members[rand.IntN(len(members))]
		for j == exclude {
			j = members[rand.IntN(len(members))]
		}
		return monkeys[j]
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := globalLeaderValue

		// Local Leader Phase
		for g, members := range groups {
			for _, i := range members {
				r := randomFromGroupExcept(g, i)
				candidate := make([]float64, dimensions)
				for d := range candidate {
					candidate[d] = monkeys[i][d] + rand.Float64()*(localLeader[g][d]-monkeys[i][d]) + rand.Float64()*(r[d]-monkeys[i][d])
					candidate[d] = clamp(candidate[d], spiderMonkeyRangeLow, spiderMonkeyRangeHigh)
				}
				candidateValue := fn.Evaluate(candidate)
				if candidateValue < values[i] {
					monkeys[i] = candidate
					values[i] = candidateValue
				}
			}
		}

		// Global Leader Phase — verovatnoća učešća proporcionalna fitnesu (bolji = veća šansa)
		maxFitness := 0.0
		fitness := make([]float64, population)
		for i, v := range values {
			if v >= 0 {
				fitness[i] = 1 / (1 + v)
			} else {
				fitness[i] = 1 + math.Abs(v)
			}
			if fitness[i] > maxFitness {
				maxFitness = fitness[i]
			}
		}

		for g, members := range groups {
			for _, i := range members {
				prob := 0.1
				if maxFitness > 0 {
					prob = 0.9*(fitness[i]/maxFitness) + 0.1
				}
				if rand.Float64() >= prob {
					continue
				}
				r := randomFromGroupExcept(g, i)
				candidate := make([]float64, dimensions)
				for d := range candidate {
					candidate[d] = monkeys[i][d] + rand.Float64()*(globalLeader[d]-monkeys[i][d]) + rand.Float64()*(r[d]-monkeys[i][d])
					candidate[d] = clamp(candidate[d], spiderMonkeyRangeLow, spiderMonkeyRangeHigh)
				}
				candidateValue := fn.Evaluate(candidate)
				if candidateValue < values[i] {
					monkeys[i] = candidate
					values[i] = candidateValue
				}
			}
		}

		for g := range groups {
			updateLocalLeader(g)
		}
		updateGlobalLeader()

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, globalLeader, globalLeaderValue, false)
		}

		if math.Abs(globalLeaderValue-prevBestValue) < tolerance {
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
		trajectory = recordTrajectory(trajectory, steps, globalLeader, globalLeaderValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "spider_monkey",
		Point:      globalLeader,
		Value:      globalLeaderValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
