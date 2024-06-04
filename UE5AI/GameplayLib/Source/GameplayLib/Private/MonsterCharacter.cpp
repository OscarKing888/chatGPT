// MonsterCharacter.cpp

#include "MonsterCharacter.h"
#include "AIController.h"
#include "Components/CapsuleComponent.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "NavigationSystem.h"

AMonsterCharacter::AMonsterCharacter()
{
	// Set default values
	Health = 100.0f;
	SightRange = 2000.0f;
	AttackRange = 200.0f;
	MovementState = EMonsterMovementState::Idle;
}

void AMonsterCharacter::BeginPlay()
{
	Super::BeginPlay();
}

float AMonsterCharacter::TakeDamage(float DamageAmount, const FDamageEvent& DamageEvent, AController* EventInstigator, AActor* DamageCauser)
{
	float ActualDamage = Super::TakeDamage(DamageAmount, DamageEvent, EventInstigator, DamageCauser);
	Health -= ActualDamage;

	if (Health <= 0)
	{
		MovementState = EMonsterMovementState::Corpse;
		GetCharacterMovement()->DisableMovement();
		NotifyDeath();
	}

	return ActualDamage;
}

void AMonsterCharacter::NotifyDeath()
{
	AAIController* AIController = Cast<AAIController>(GetController());
	if (AIController)
	{
		UBlackboardComponent* BlackboardComponent = AIController->GetBlackboardComponent();
		if (BlackboardComponent)
		{
			BlackboardComponent->SetValueAsBool(TEXT("IsDead"), true);
		}
	}
}

void AMonsterCharacter::Attack(AActor* AttackTarget)
{
	UE_LOG(LogTemp, Warning, TEXT("Monster::Attack"));
	OnAttack(AttackTarget);
}

void AMonsterCharacter::OnAttack_Implementation(AActor* Actor)
{

}



void ACustomNavAIController::FindPathForMoveRequest(const FAIMoveRequest& MoveRequest, FPathFindingQuery& Query, FNavPathSharedPtr& OutPath) const
{
	TWeakObjectPtr<ANavigationData> PreferredNavData;
	AMonsterCharacter* Monster = Cast<AMonsterCharacter>(GetPawn());
	PreferredNavData = Monster->PreferredNavData;

	UNavigationSystemV1* NavSys = FNavigationSystem::GetCurrent<UNavigationSystemV1>(GetWorld());
	if (NavSys)
	{
		const ANavigationData* NavData = nullptr;

		// Use the specified navigation data if available
		if (PreferredNavData.IsValid())
		{
			NavData = PreferredNavData.Get();
		}
		else
		{
			NavData = MoveRequest.IsUsingPathfinding() ? NavSys->GetNavDataForProps(GetNavAgentPropertiesRef(), GetNavAgentLocation()) :
				NavSys->GetAbstractNavData();
		}

		if (NavData)
		{
			FVector GoalLocation = MoveRequest.GetGoalLocation();
			if (MoveRequest.IsMoveToActorRequest())
			{
				const INavAgentInterface* NavGoal = Cast<const INavAgentInterface>(MoveRequest.GetGoalActor());
				if (NavGoal)
				{
					const FVector Offset = NavGoal->GetMoveGoalOffset(this);
					GoalLocation = FQuatRotationTranslationMatrix(MoveRequest.GetGoalActor()->GetActorQuat(), NavGoal->GetNavAgentLocation()).TransformPosition(Offset);
				}
				else
				{
					GoalLocation = MoveRequest.GetGoalActor()->GetActorLocation();
				}
			}

			FSharedConstNavQueryFilter NavFilter = UNavigationQueryFilter::GetQueryFilter(*NavData, this, MoveRequest.GetNavigationFilter());
			Query = FPathFindingQuery(*this, *NavData, GetNavAgentLocation(), GoalLocation, NavFilter);
			Query.SetAllowPartialPaths(MoveRequest.IsUsingPartialPaths());
			Query.SetRequireNavigableEndLocation(MoveRequest.IsNavigableEndLocationRequired());
			if (MoveRequest.IsApplyingCostLimitFromHeuristic())
			{
				const float HeuristicScale = NavFilter->GetHeuristicScale();
				Query.CostLimit = FPathFindingQuery::ComputeCostLimitFromHeuristic(Query.StartLocation, Query.EndLocation, HeuristicScale, MoveRequest.GetCostLimitFactor(), MoveRequest.GetMinimumCostLimit());
			}

// 			if (PathFollowingComponent)
// 			{
// 				PathFollowingComponent->OnPathfindingQuery(Query);
// 			}

			FPathFindingResult PathResult = NavSys->FindPathSync(Query);
			if (PathResult.Result != ENavigationQueryResult::Error)
			{
				if (PathResult.IsSuccessful() && PathResult.Path.IsValid())
				{
					if (MoveRequest.IsMoveToActorRequest())
					{
						PathResult.Path->SetGoalActorObservation(*MoveRequest.GetGoalActor(), 100.0f);
					}

					PathResult.Path->EnableRecalculationOnInvalidation(true);
					OutPath = PathResult.Path;
				}
			}
			else
			{
				UE_VLOG(this, LogTemp, Error, TEXT("ACustomNavAIController Trying to find path to %s resulted in Error")
					, MoveRequest.IsMoveToActorRequest() ? *GetNameSafe(MoveRequest.GetGoalActor()) : *MoveRequest.GetGoalLocation().ToString());
				UE_VLOG_SEGMENT(this, LogTemp, Error, GetPawn() ? GetPawn()->GetActorLocation() : FAISystem::InvalidLocation
					, MoveRequest.GetGoalLocation(), FColor::Red, TEXT("Failed move to %s"), *GetNameSafe(MoveRequest.GetGoalActor()));
			}
		}
	}
}
