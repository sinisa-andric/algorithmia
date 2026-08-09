package route

import (
	"algorithmia/src/defaults"
	"algorithmia/src/docs"
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"algorithmia/src/solvers"
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

type AlgorithmiaResponse struct {
	Approved bool        `json:"approved"`
	Message  string      `json:"message,omitempty"`
	Response interface{} `json:"response,omitempty"`
}

// DefaultsHandler vraća podrazumevani payload za svaki solver metod
func DefaultsHandler(ctx *gin.Context) {
	ctx.JSON(http.StatusOK, gin.H{
		"approved": true,
		"message":  "solver defaults",
		"response": defaults.SolverDefaults,
	})
}

// FunctionsHandler vraća dokumentaciju za svaku benchmark funkciju
func FunctionsHandler(ctx *gin.Context) {
	ctx.JSON(http.StatusOK, gin.H{
		"approved": true,
		"message":  "available functions",
		"response": docs.FunctionDocs,
	})
}

// SolversHandler vraća dokumentaciju za svaki solver
func SolversHandler(ctx *gin.Context) {
	ctx.JSON(http.StatusOK, gin.H{
		"approved": true,
		"message":  "available solvers",
		"response": docs.SolverDocs,
	})
}

func FunctionHandler(ctx *gin.Context) {
	name := ctx.Param("name")
	fn, ok := docs.FunctionDocs[name]
	if !ok {
		ctx.JSON(http.StatusNotFound, gin.H{
			"approved": false,
			"message":  "function not found: " + name,
		})
		return
	}
	ctx.JSON(http.StatusOK, gin.H{
		"approved": true,
		"message":  "function details",
		"response": fn,
	})
}

func SolverHandler(ctx *gin.Context) {
	name := ctx.Param("name")
	solver, ok := docs.SolverDocs[name]
	if !ok {
		ctx.JSON(http.StatusNotFound, gin.H{
			"approved": false,
			"message":  "solver not found: " + name,
		})
		return
	}
	ctx.JSON(http.StatusOK, gin.H{
		"approved": true,
		"message":  "solver details",
		"response": solver,
	})
}

func SolveProblemHandler(ctx *gin.Context) {
	log := zap.New(nil)
	defer log.Sync()
	requestId := ctx.GetString("id")
	log = log.Named("[Algorithmia:ProblemHandler]").WithOptions(
		zap.Fields(
			zap.String("requestId", requestId),
		),
	)
	log.Info("solve problem started")

	var problem models.Problem
	if err := ctx.BindJSON(&problem); err != nil {
		log.Error("failed to bind request body", zap.Error(err))
		ctx.JSON(
			http.StatusBadRequest,
			AlgorithmiaResponse{
				Approved: false,
				Message:  "failed to bind request body: " + err.Error(),
			},
		)
		return
	}

	result, err := SolveProblem(problem)
	if err != nil {
		ctx.JSON(
			http.StatusNotAcceptable,
			AlgorithmiaResponse{
				Approved: false,
				Message:  "not solved: " + err.Error(),
			},
		)
		return
	}

	ctx.JSON(
		http.StatusCreated,
		AlgorithmiaResponse{
			Approved: true,
			Message:  "solved!",
			Response: result,
		},
	)
	log.Info("solve problem finished",
		zap.String("method", problem.Method),
		zap.Any("point", problem.Point),
	)
}

func SolveProblem(problem models.Problem) (result models.Result, err error) {

	fnName, _ := problem.Payload["function"].(string)
	if _, err := functions.Get(fnName); err != nil {
		return result, err
	}

	problem.Payload = defaults.MergeWithDefaults(problem.Method, problem.Payload)

	switch problem.Method {
	case "gradient_descent":
		return solvers.GradientDescent(problem)
	case "golden_section":
		return solvers.GoldenSection(problem)
	case "newton":
		return solvers.Newton(problem)
	case "random_point", "":
		return solvers.RandomPoint(problem)
	case "simulated_annealing":
		return solvers.SimulatedAnnealing(problem)
	case "coordinate_descent":
		return solvers.CoordinateDescent(problem)
	case "conjugate_gradient":
		return solvers.ConjugateGradient(problem)
	case "nelder_mead":
		return solvers.NelderMead(problem)
	case "hill_climbing":
		return solvers.HillClimbing(problem)
	case "pattern_search":
		return solvers.PatternSearch(problem)
	case "quadratic_interpolation":
		return solvers.QuadraticInterpolation(problem)
	case "subgradient":
		return solvers.Subgradient(problem)
	case "tabu_search":
		return solvers.TabuSearch(problem)
	case "particle_swarm":
		return solvers.ParticleSwarm(problem)
	case "genetic_algorithm":
		return solvers.GeneticAlgorithm(problem)
	case "differential_evolution":
		return solvers.DifferentialEvolution(problem)
	case "quasi_newton":
		return solvers.QuasiNewton(problem)
	case "mirror_descent":
		return solvers.MirrorDescent(problem)
	case "random_search":
		return solvers.RandomSearch(problem)
	case "stochastic_tunneling":
		return solvers.StochasticTunneling(problem)
	case "memetic_algorithm":
		return solvers.MemeticAlgorithm(problem)
	case "frank_wolfe":
		return solvers.FrankWolfe(problem)
	case "spsa":
		return solvers.SPSA(problem)
	case "bundle_method":
		return solvers.BundleMethod(problem)
	case "ellipsoid_method":
		return solvers.EllipsoidMethod(problem)
	case "dynamic_relaxation":
		return solvers.DynamicRelaxation(problem)
	case "cma_es":
		return solvers.CmaEs(problem)
	case "bayesian_optimization":
		return solvers.BayesianOptimization(problem)
	case "firefly_algorithm":
		return solvers.FireflyAlgorithm(problem)
	case "ant_colony":
		return solvers.AntColony(problem)
	case "harmony_search":
		return solvers.HarmonySearch(problem)
	case "artificial_bee_colony":
		return solvers.ArtificialBeeColony(problem)
	case "polyak_subgradient":
		return solvers.PolyakSubgradient(problem)
	case "proximal_gradient":
		return solvers.ProximalGradient(problem)
	case "grey_wolf":
		return solvers.GreyWolf(problem)
	case "whale_optimization":
		return solvers.WhaleOptimization(problem)
	case "bat_algorithm":
		return solvers.BatAlgorithm(problem)
	case "cuckoo_search":
		return solvers.CuckooSearch(problem)
	case "evolution_strategy":
		return solvers.EvolutionStrategy(problem)
	case "self_adaptive_es":
		return solvers.SelfAdaptiveES(problem)
	case "direct_search":
		return solvers.DirectSearch(problem)
	case "mads":
		return solvers.MADS(problem)
	case "grasshopper":
		return solvers.Grasshopper(problem)
	case "dragonfly":
		return solvers.Dragonfly(problem)
	case "salp_swarm":
		return solvers.SalpSwarm(problem)
	case "spsa_bundle":
		return solvers.SPSABundle(problem)
	case "nelder_mead_adaptive":
		return solvers.NelderMeadAdaptive(problem)
	case "gravitational_search":
		return solvers.GravitationalSearch(problem)
	case "charged_system_search":
		return solvers.ChargedSystemSearch(problem)
	case "big_bang_big_crunch":
		return solvers.BigBangBigCrunch(problem)
	case "flower_pollination":
		return solvers.FlowerPollination(problem)
	case "basin_hopping":
		return solvers.BasinHopping(problem)
	case "iterated_local_search":
		return solvers.IteratedLocalSearch(problem)
	case "monarch_butterfly":
		return solvers.MonarchButterfly(problem)
	case "slime_mould":
		return solvers.SlimeMould(problem)
	case "aquila":
		return solvers.Aquila(problem)
	case "nelder_mead_restart":
		return solvers.NelderMeadRestart(problem)
	case "pso_gd":
		return solvers.PSOGD(problem)
	case "ga_local_search":
		return solvers.GALocalSearch(problem)
	case "sgd":
		return solvers.Sgd(problem)
	case "minibatch_gd":
		return solvers.MinibatchGd(problem)
	case "momentum":
		return solvers.Momentum(problem)
	case "nag":
		return solvers.Nag(problem)
	case "adagrad":
		return solvers.Adagrad(problem)
	case "rmsprop":
		return solvers.Rmsprop(problem)
	case "adadelta":
		return solvers.Adadelta(problem)
	case "adam":
		return solvers.Adam(problem)
	case "nadam":
		return solvers.Nadam(problem)
	case "adamw":
		return solvers.Adamw(problem)
	case "radam":
		return solvers.Radam(problem)
	case "amsgrad":
		return solvers.Amsgrad(problem)
	case "adamax":
		return solvers.Adamax(problem)
	case "asgd":
		return solvers.Asgd(problem)
	case "rprop":
		return solvers.Rprop(problem)
	case "irprop_plus":
		return solvers.IrpropPlus(problem)
	case "sign_sgd":
		return solvers.SignSgd(problem)
	case "adabound":
		return solvers.Adabound(problem)
	case "diffgrad":
		return solvers.Diffgrad(problem)
	case "lookahead":
		return solvers.Lookahead(problem)
	case "lbfgs":
		return solvers.Lbfgs(problem)
	case "barzilai_borwein":
		return solvers.BarzilaiBorwein(problem)
	case "polak_ribiere_cg":
		return solvers.PolakRibiereCg(problem)
	case "hestenes_stiefel_cg":
		return solvers.HestenesStiefelCg(problem)
	case "dai_yuan_cg":
		return solvers.DaiYuanCg(problem)
	case "levenberg_marquardt":
		return solvers.LevenbergMarquardt(problem)
	case "trust_region_dogleg":
		return solvers.TrustRegionDogleg(problem)
	case "augmented_lagrangian":
		return solvers.AugmentedLagrangian(problem)
	case "brent":
		return solvers.Brent(problem)
	case "fibonacci_search":
		return solvers.FibonacciSearch(problem)
	case "dichotomous_search":
		return solvers.DichotomousSearch(problem)
	case "secant_method":
		return solvers.SecantMethod(problem)
	case "gauss_newton":
		return solvers.GaussNewton(problem)
	case "rosenbrock_method":
		return solvers.RosenbrockMethod(problem)
	case "compass_search":
		return solvers.CompassSearch(problem)
	case "adamo":
		return solvers.AdamO(problem)
	case "glider_snake_optimizer":
		return solvers.GliderSnakeOptimizer(problem)
	case "birds_of_paradise":
		return solvers.BirdsOfParadise(problem)
	case "harris_hawks":
		return solvers.HarrisHawks(problem)
	case "marine_predators":
		return solvers.MarinePredators(problem)
	case "dung_beetle":
		return solvers.DungBeetle(problem)
	case "secretary_bird":
		return solvers.SecretaryBird(problem)
	case "golden_eagle":
		return solvers.GoldenEagle(problem)
	case "artificial_gorilla":
		return solvers.ArtificialGorilla(problem)
	case "pelican":
		return solvers.Pelican(problem)
	case "fennec_fox":
		return solvers.FennecFox(problem)
	case "coati":
		return solvers.Coati(problem)
	case "mountain_gazelle":
		return solvers.MountainGazelle(problem)
	case "honey_badger":
		return solvers.HoneyBadger(problem)
	case "golden_jackal":
		return solvers.GoldenJackal(problem)
	case "zebra":
		return solvers.Zebra(problem)
	case "cheetah":
		return solvers.Cheetah(problem)
	case "sand_cat":
		return solvers.SandCat(problem)
	case "african_vulture":
		return solvers.AfricanVulture(problem)
	case "remora":
		return solvers.Remora(problem)
	case "beluga_whale":
		return solvers.BelugaWhale(problem)
	case "white_shark":
		return solvers.WhiteShark(problem)
	case "jellyfish":
		return solvers.Jellyfish(problem)
	case "moth_flame":
		return solvers.MothFlame(problem)
	case "crow_search":
		return solvers.CrowSearch(problem)
	case "krill_herd":
		return solvers.KrillHerd(problem)
	case "antlion":
		return solvers.Antlion(problem)
	case "tunicate":
		return solvers.Tunicate(problem)
	case "mayfly":
		return solvers.Mayfly(problem)
	case "red_deer":
		return solvers.RedDeer(problem)
	case "seagull":
		return solvers.Seagull(problem)
	case "dingo":
		return solvers.Dingo(problem)
	case "barnacles":
		return solvers.Barnacles(problem)
	case "sailfish":
		return solvers.Sailfish(problem)
	case "naked_mole_rat":
		return solvers.NakedMoleRat(problem)
	case "chimp":
		return solvers.Chimp(problem)
	case "sooty_tern":
		return solvers.SootyTern(problem)
	case "sea_lion":
		return solvers.SeaLion(problem)
	case "shuffled_frog":
		return solvers.ShuffledFrog(problem)
	case "cat_swarm":
		return solvers.CatSwarm(problem)
	case "glowworm":
		return solvers.Glowworm(problem)
	case "fruit_fly":
		return solvers.FruitFly(problem)
	case "pigeon":
		return solvers.Pigeon(problem)
	case "spider_monkey":
		return solvers.SpiderMonkey(problem)
	case "elephant":
		return solvers.Elephant(problem)
	case "moth_search":
		return solvers.MothSearch(problem)
	case "crested_porcupine":
		return solvers.CrestedPorcupine(problem)
	case "ostrich":
		return solvers.Ostrich(problem)
	case "nutcracker":
		return solvers.Nutcracker(problem)
	case "koala":
		return solvers.Koala(problem)
	case "peregrine_falcon":
		return solvers.PeregrineFalcon(problem)
	case "meerkat":
		return solvers.Meerkat(problem)
	case "platypus":
		return solvers.Platypus(problem)
	case "chameleon":
		return solvers.Chameleon(problem)
	case "arctic_fox":
		return solvers.ArcticFox(problem)
	case "flying_foxes":
		return solvers.FlyingFoxes(problem)
	case "kangaroo":
		return solvers.Kangaroo(problem)
	case "flamingo":
		return solvers.Flamingo(problem)
	case "tarantula_hawk":
		return solvers.TarantulaHawk(problem)
	case "swallow":
		return solvers.Swallow(problem)
	case "termite_colony":
		return solvers.TermiteColony(problem)
	case "wasp":
		return solvers.Wasp(problem)
	case "mosquito":
		return solvers.Mosquito(problem)
	case "bacterial_foraging":
		return solvers.BacterialForaging(problem)
	case "pig":
		return solvers.Pig(problem)
	case "horse":
		return solvers.Horse(problem)
	case "donkey":
		return solvers.Donkey(problem)
	case "camel":
		return solvers.Camel(problem)
	case "sheep":
		return solvers.Sheep(problem)
	case "goat":
		return solvers.Goat(problem)
	case "duck":
		return solvers.Duck(problem)
	case "goose":
		return solvers.Goose(problem)
	case "penguin_search":
		return solvers.PenguinSearch(problem)
	case "owl":
		return solvers.Owl(problem)
	case "woodpecker":
		return solvers.Woodpecker(problem)
	case "parrot":
		return solvers.Parrot(problem)
	case "eagle_strategy":
		return solvers.EagleStrategy(problem)
	case "goshawk":
		return solvers.Goshawk(problem)
	case "saker_falcon":
		return solvers.SakerFalcon(problem)
	case "shark_smell":
		return solvers.SharkSmell(problem)
	case "dolphin":
		return solvers.Dolphin(problem)
	case "sea_turtle":
		return solvers.SeaTurtle(problem)
	case "octopus":
		return solvers.Octopus(problem)
	case "starfish":
		return solvers.Starfish(problem)
	case "crab":
		return solvers.Crab(problem)
	case "lobster":
		return solvers.Lobster(problem)
	case "shrimp":
		return solvers.Shrimp(problem)
	case "coral_reef":
		return solvers.CoralReef(problem)
	case "sea_anemone":
		return solvers.SeaAnemone(problem)
	case "cockroach":
		return solvers.Cockroach(problem)
	case "vampire_bat":
		return solvers.VampireBat(problem)
	case "manta_ray":
		return solvers.MantaRay(problem)
	case "sea_horse":
		return solvers.SeaHorse(problem)
	case "puffer_fish":
		return solvers.PufferFish(problem)
	case "electric_eel":
		return solvers.ElectricEel(problem)
	case "barracuda":
		return solvers.Barracuda(problem)
	case "squid":
		return solvers.Squid(problem)
	case "sea_urchin":
		return solvers.SeaUrchin(problem)
	case "mals":
		return solvers.MALS(problem)
	case "fpso":
		return solvers.FPSO(problem)
	case "sgha":
		return solvers.SGHA(problem)
	case "sa_ts":
		return solvers.SaTs(problem)
	case "aco_abc":
		return solvers.AcoAbc(problem)
	case "gehm":
		return solvers.GEHM(problem)
	case "clonalg":
		return solvers.Clonalg(problem)
	case "negative_selection":
		return solvers.NegativeSelection(problem)
	case "dendritic_cell":
		return solvers.DendriticCell(problem)
	case "airs":
		return solvers.Airs(problem)
	case "immune_network":
		return solvers.ImmuneNetwork(problem)
	case "t_cell":
		return solvers.TCell(problem)
	case "humoral_immune":
		return solvers.HumoralImmune(problem)
	case "leukocyte":
		return solvers.Leukocyte(problem)
	case "antibody_clonal":
		return solvers.AntibodyClonal(problem)
	case "tissue_layer":
		return solvers.TissueLayer(problem)
	case "invasive_weed":
		return solvers.InvasiveWeed(problem)
	case "paddy_field":
		return solvers.PaddyField(problem)
	case "root_growth":
		return solvers.RootGrowth(problem)
	case "tree_growth":
		return solvers.TreeGrowth(problem)
	case "saplings":
		return solvers.Saplings(problem)
	case "runner_root":
		return solvers.RunnerRoot(problem)
	case "plant_propagation":
		return solvers.PlantPropagation(problem)
	case "seed_swarm":
		return solvers.SeedSwarm(problem)
	case "forest_optimization":
		return solvers.ForestOptimization(problem)
	case "bamboo_growth":
		return solvers.BambooGrowth(problem)
	case "cactus_search":
		return solvers.CactusSearch(problem)
	case "weed_colony":
		return solvers.WeedColony(problem)
	case "photosynthesis":
		return solvers.Photosynthesis(problem)
	case "sunflower":
		return solvers.Sunflower(problem)
	case "sine_cosine":
		return solvers.SineCosine(problem)
	case "equilibrium_optimizer":
		return solvers.EquilibriumOptimizer(problem)
	case "multiverse":
		return solvers.Multiverse(problem)
	case "water_cycle":
		return solvers.WaterCycle(problem)
	case "atom_search":
		return solvers.AtomSearch(problem)
	case "henry_gas":
		return solvers.HenryGas(problem)
	case "lightning_search":
		return solvers.LightningSearch(problem)
	case "wind_driven":
		return solvers.WindDriven(problem)
	case "volcano_eruption":
		return solvers.VolcanoEruption(problem)
	case "rain_water":
		return solvers.RainWater(problem)
	case "mine_blast":
		return solvers.MineBlast(problem)
	case "river_formation":
		return solvers.RiverFormation(problem)
	case "galaxy_search":
		return solvers.GalaxySearch(problem)
	case "nuclear_reaction":
		return solvers.NuclearReaction(problem)
	case "electromagnetism":
		return solvers.Electromagnetism(problem)
	case "heat_transfer":
		return solvers.HeatTransfer(problem)
	case "thermal_exchange":
		return solvers.ThermalExchange(problem)
	case "optics_inspired":
		return solvers.OpticsInspired(problem)
	case "light_spectrum":
		return solvers.LightSpectrum(problem)
	case "water_evaporation":
		return solvers.WaterEvaporation(problem)
	case "kinetic_energy":
		return solvers.KineticEnergy(problem)
	case "ion_motion":
		return solvers.IonMotion(problem)
	case "electrostatic_discharge":
		return solvers.ElectrostaticDischarge(problem)
	case "gravitational_local_search":
		return solvers.GravitationalLocalSearch(problem)
	case "central_force":
		return solvers.CentralForce(problem)
	case "vortex_search":
		return solvers.VortexSearch(problem)
	case "spiral_dynamics":
		return solvers.SpiralDynamics(problem)
	case "crystal_structure":
		return solvers.CrystalStructure(problem)
	case "tlbo":
		return solvers.Tlbo(problem)
	case "imperialist_competitive":
		return solvers.ImperialistCompetitive(problem)
	case "soccer_league":
		return solvers.SoccerLeague(problem)
	case "cricket_behavior":
		return solvers.CricketBehavior(problem)
	case "parliamentary":
		return solvers.Parliamentary(problem)
	case "ideology_algorithm":
		return solvers.IdeologyAlgorithm(problem)
	case "league_championship":
		return solvers.LeagueChampionship(problem)
	case "football_optimization":
		return solvers.FootballOptimization(problem)
	case "basketball_optimization":
		return solvers.BasketballOptimization(problem)
	case "tennis_optimization":
		return solvers.TennisOptimization(problem)
	case "chess_optimization":
		return solvers.ChessOptimization(problem)
	case "war_strategy":
		return solvers.WarStrategy(problem)
	case "election_optimization":
		return solvers.ElectionOptimization(problem)
	case "voting_optimization":
		return solvers.VotingOptimization(problem)
	case "colonial_competitive":
		return solvers.ColonialCompetitive(problem)
	case "human_mental_search":
		return solvers.HumanMentalSearch(problem)
	case "social_network":
		return solvers.SocialNetwork(problem)
	case "sports_league":
		return solvers.SportsLeague(problem)
	case "team_game_algorithm":
		return solvers.TeamGameAlgorithm(problem)
	case "exam_invigilation":
		return solvers.ExamInvigilation(problem)
	case "driving_training":
		return solvers.DrivingTraining(problem)
	case "student_psychology":
		return solvers.StudentPsychology(problem)
	case "queen_bee_evolution":
		return solvers.QueenBeeEvolution(problem)
	case "brainstorming_optimization":
		return solvers.BrainstormingOptimization(problem)
	case "political_optimizer":
		return solvers.PoliticalOptimizer(problem)
	case "gaining_sharing":
		return solvers.GainingSharing(problem)
	case "cultural_algorithm":
		return solvers.CulturalAlgorithm(problem)
	case "commerce_algorithm":
		return solvers.CommerceAlgorithm(problem)
	case "fista":
		return solvers.Fista(problem)
	case "admm":
		return solvers.Admm(problem)
	case "linearized_admm":
		return solvers.LinearizedAdmm(problem)
	case "douglas_rachford":
		return solvers.DouglasRachford(problem)
	case "peaceman_rachford":
		return solvers.PeacemanRachford(problem)
	case "davis_yin":
		return solvers.DavisYin(problem)
	case "pdhg_chambolle_pock":
		return solvers.PdhgChambollePock(problem)
	case "split_bregman":
		return solvers.SplitBregman(problem)
	case "proximal_point":
		return solvers.ProximalPoint(problem)
	case "palm":
		return solvers.Palm(problem)
	case "forward_backward_ls":
		return solvers.ForwardBackwardLs(problem)
	case "inertial_proximal":
		return solvers.InertialProximal(problem)
	case "majorization_minimization":
		return solvers.MajorizationMinimization(problem)
	case "proximal_newton":
		return solvers.ProximalNewton(problem)
	case "proximal_bundle":
		return solvers.ProximalBundle(problem)
	case "relaxed_admm":
		return solvers.RelaxedAdmm(problem)
	case "proximal_gradient_bb":
		return solvers.ProximalGradientBb(problem)
	case "alternating_minimization":
		return solvers.AlternatingMinimization(problem)
	case "bregman_proximal_gradient":
		return solvers.BregmanProximalGradient(problem)
	case "mirror_prox":
		return solvers.MirrorProx(problem)
	case "proximal_admm":
		return solvers.ProximalAdmm(problem)
	case "semismooth_newton":
		return solvers.SemismoothNewton(problem)
	case "dykstra_algorithm":
		return solvers.DykstraAlgorithm(problem)
	case "proximal_coordinate_descent":
		return solvers.ProximalCoordinateDescent(problem)
	case "halpern_iteration":
		return solvers.HalpernIteration(problem)
	case "nesterov_smoothing":
		return solvers.NesterovSmoothing(problem)
	case "optimal_gradient_method":
		return solvers.OptimalGradientMethod(problem)
	case "auslender_teboulle":
		return solvers.AuslenderTeboulle(problem)
	case "lion":
		return solvers.Lion(problem)
	case "adabelief":
		return solvers.Adabelief(problem)
	case "sag":
		return solvers.Sag(problem)
	case "svrg":
		return solvers.Svrg(problem)
	case "yogi":
		return solvers.Yogi(problem)
	case "qhadam":
		return solvers.Qhadam(problem)
	case "qhm":
		return solvers.Qhm(problem)
	case "ftrl":
		return solvers.Ftrl(problem)
	case "adan":
		return solvers.Adan(problem)
	case "prodigy":
		return solvers.Prodigy(problem)
	case "dadaptation":
		return solvers.Dadaptation(problem)
	case "sm3":
		return solvers.Sm3(problem)
	case "adafactor":
		return solvers.Adafactor(problem)
	case "novograd":
		return solvers.Novograd(problem)
	case "lamb":
		return solvers.Lamb(problem)
	case "lars":
		return solvers.Lars(problem)
	case "madgrad":
		return solvers.Madgrad(problem)
	case "apollo":
		return solvers.Apollo(problem)
	case "quickprop":
		return solvers.Quickprop(problem)
	case "padam":
		return solvers.Padam(problem)
	case "swats":
		return solvers.Swats(problem)
	case "ranger":
		return solvers.Ranger(problem)
	case "amsbound":
		return solvers.Amsbound(problem)
	case "radambound":
		return solvers.Radambound(problem)
	case "storm":
		return solvers.Storm(problem)
	case "adanw":
		return solvers.Adanw(problem)
	case "schedule_free_adam":
		return solvers.ScheduleFreeAdam(problem)
	case "schedule_free_sgd":
		return solvers.ScheduleFreeSgd(problem)
	case "grokfast":
		return solvers.Grokfast(problem)
	case "aggmo":
		return solvers.Aggmo(problem)
	case "sophia":
		return solvers.Sophia(problem)
	case "adahessian":
		return solvers.Adahessian(problem)
	case "fromage":
		return solvers.Fromage(problem)
	case "shampoo":
		return solvers.Shampoo(problem)
	case "adai":
		return solvers.Adai(problem)
	case "came":
		return solvers.Came(problem)
	case "nero":
		return solvers.Nero(problem)
	case "signum":
		return solvers.Signum(problem)
	case "nadamw":
		return solvers.Nadamw(problem)
	case "larc":
		return solvers.Larc(problem)
	case "pid_optimizer":
		return solvers.PidOptimizer(problem)
	case "ada_momentum":
		return solvers.AdaMomentum(problem)
	case "dfp":
		return solvers.Dfp(problem)
	case "sr1":
		return solvers.Sr1(problem)
	case "truncated_newton":
		return solvers.TruncatedNewton(problem)
	case "trust_region_reflective":
		return solvers.TrustRegionReflective(problem)
	case "trust_region_subspace":
		return solvers.TrustRegionSubspace(problem)
	case "powell_conjugate_direction":
		return solvers.PowellConjugateDirection(problem)
	case "box_complex_method":
		return solvers.BoxComplexMethod(problem)
	case "torczon_multidirectional":
		return solvers.TorczonMultidirectional(problem)
	case "generating_set_search":
		return solvers.GeneratingSetSearch(problem)
	case "successive_quadratic_approximation":
		return solvers.SuccessiveQuadraticApproximation(problem)
	case "grid_search":
		return solvers.GridSearch(problem)
	case "halton_sequence_search":
		return solvers.HaltonSequenceSearch(problem)
	case "sobol_sequence_search":
		return solvers.SobolSequenceSearch(problem)
	case "latin_hypercube_search":
		return solvers.LatinHypercubeSearch(problem)
	case "frank_wolfe_box":
		return solvers.FrankWolfeBox(problem)
	case "active_set_bound":
		return solvers.ActiveSetBound(problem)
	default:
		err = fmt.Errorf("unknown method: %s", problem.Method)
		return result, err
	}
}
