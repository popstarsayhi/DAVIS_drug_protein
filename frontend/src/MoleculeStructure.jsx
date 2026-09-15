import { useEffect, useRef } from "react";
import SmilesDrawer from "smiles-drawer";

function MoleculeStructure({ smiles }) {
    const canvasRef = useRef(null);

    useEffect(() => {
        if (!smiles || !canvasRef.current) {
            return;
        }

        const drawer = new SmilesDrawer.Drawer({
            width: 260,
            height: 180,
            bondThickness: 1.5,
            padding: 15,
        });

        SmilesDrawer.parse(
            smiles,
            (tree) => {
                drawer.draw(
                    tree,
                    canvasRef.current,
                    "light",
                    false
                );
            },
            (error) => {
                console.error(
                    "Could not draw molecule:",
                    smiles,
                    error
                );
            }
        );
    }, [smiles]);

    return (
        <canvas
            ref={canvasRef}
            width="260"
            height="180"
        />
    );
}

export default MoleculeStructure;