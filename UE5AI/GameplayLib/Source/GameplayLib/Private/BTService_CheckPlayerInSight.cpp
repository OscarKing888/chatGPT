// BTService_CheckPlayerInSight.cpp

#include "BTService_CheckPlayerInSight.h"
#include "MonsterCharacter.h"
#include "AIController.h"
#include "Kismet/GameplayStatics.h"
#include "BehaviorTree/BlackboardComponent.h"

#if UE_BUILD_DEVELOPMENT
static TAutoConsoleVariable<int32> CVarShowMonsterLineCheck(
    TEXT("r.GameLib.ShowMonsterLineCheck"),
    0,
    TEXT("Show monster line check.\n")
    TEXT("0: Disabled, 1: Enabled"),
    ECVF_Default);
#endif

UBTService_CheckPlayerInSight::UBTService_CheckPlayerInSight()
{
    NodeName = TEXT("Check Player In Sight");
}

void UBTService_CheckPlayerInSight::TickNode(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds)
{
    Super::TickNode(OwnerComp, NodeMemory, DeltaSeconds);

    AAIController* AIController = OwnerComp.GetAIOwner();
    if (AIController == nullptr)
    {
        return;
    }

    AMonsterCharacter* Monster = Cast<AMonsterCharacter>(AIController->GetPawn());
    if (Monster == nullptr)
    {
        return;
    }

    UBlackboardComponent* BlackboardComponent = AIController->GetBlackboardComponent();
    if (BlackboardComponent == nullptr)
    {
        return;
    }

    APlayerController* PlayerController = UGameplayStatics::GetPlayerController(GetWorld(), 0);
    if (PlayerController != nullptr)
    {
        APawn* PlayerPawn = PlayerController->GetPawn();
        if (PlayerPawn != nullptr)
        {
            FVector PlayerLocation = PlayerPawn->GetActorLocation();
            FVector MonsterLocation = Monster->GetActorLocation();
            FVector DirectionToPlayer = PlayerLocation - MonsterLocation;

            float DistanceToPlayer = FVector::Distance(PlayerLocation, MonsterLocation);

            // Check if the player is within the monster's sight range
            float SightRange = Monster->SightRange;
            if (DistanceToPlayer <= SightRange)
            {
                // Perform a line trace to check if the monster has a clear line of sight to the player
                FHitResult HitResult;
                FCollisionQueryParams QueryParams;
                QueryParams.AddIgnoredActor(Monster);

                bool bHit = GetWorld()->LineTraceSingleByChannel(HitResult, MonsterLocation, PlayerLocation, ECC_Visibility, QueryParams);

#if UE_BUILD_DEVELOPMENT
                extern TAutoConsoleVariable<int32> CVarShowMonsterLineCheck;
                if (CVarShowMonsterLineCheck.GetValueOnGameThread() != 0)
                {
                    DrawDebugLine(GetWorld(), MonsterLocation, PlayerLocation, bHit ? FColor::Red : FColor::Green, false, 5.0f);
                    //DrawDebugSphere(GetWorld(), GetActorLocation(), SpawnRadius, 24, FColor::Green, false, -1.f, 0, 1.f);
                }
#endif

                AActor* HitActor = HitResult.GetActor();
                if (!bHit || HitActor == PlayerPawn)
                {
                    // Player is in sight, update the blackboard key
                    BlackboardComponent->SetValueAsObject(GetSelectedBlackboardKey(), HitActor);
                    return;
                }
            }
        }
    }

    // If the player is not in sight, update the blackboard key
    BlackboardComponent->SetValueAsObject(GetSelectedBlackboardKey(), nullptr);
}
