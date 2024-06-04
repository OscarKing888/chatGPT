// MonsterAIController.cpp

#include "MonsterAIController.h"
#include "MonsterCharacter.h"
#include "BehaviorTree/BehaviorTree.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "BehaviorTree/Blackboard/BlackboardKeyType_Enum.h"
#include "BehaviorTree/Blackboard/BlackboardKeyType_Vector.h"
#include "Kismet/GameplayStatics.h"

AMonsterAIController::AMonsterAIController()
{
    BlackboardComponent = CreateDefaultSubobject<UBlackboardComponent>(TEXT("BlackboardComponent"));

    // 这里替换为你创建的行为树资源的路径
	/*static ConstructorHelpers::FObjectFinder<UBehaviorTree> BehaviorTreeAsset(TEXT("BehaviorTree'/Game/AI/MonsterBehaviorTree.MonsterBehaviorTree'"));
	if (BehaviorTreeAsset.Succeeded())
	{
		BehaviorTree = BehaviorTreeAsset.Object;
	}*/
}

void AMonsterAIController::OnPossess(APawn* InPawn)
{
    Super::OnPossess(InPawn);

    AMonsterCharacter* MonsterCharacter = Cast<AMonsterCharacter>(InPawn);
    if (MonsterCharacter && BehaviorTree)
    {
        // Initialize the Blackboard values
        BlackboardComponent->InitializeBlackboard(*BehaviorTree->BlackboardAsset);

        // Start running the behavior tree
        RunBehaviorTree(BehaviorTree);
    }
}

void AMonsterAIController::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    // Check if the Monster is in Corpse state
    AMonsterCharacter* Monster = Cast<AMonsterCharacter>(GetPawn());
    if (!Monster || Monster->GetMovementState() == EMonsterMovementState::Corpse)
    {
        // Do not respond to movement input in Corpse state
        return;
    }

    // You can update the Blackboard keys as needed here.
    // For example, if you want to update the player's location in the Blackboard
    APlayerController* PlayerController = UGameplayStatics::GetPlayerController(GetWorld(), 0);
    if (PlayerController != nullptr)
    {
        FVector PlayerLocation = Monster->GetActorLocation();
        BlackboardComponent->SetValueAsVector(TEXT("PlayerLocation"), PlayerLocation);
    }
}

